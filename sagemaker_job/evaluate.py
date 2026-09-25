#!/usr/bin/env python3
"""SageMaker Job entrypoint: score ONNX models on a rewritten eval CSV."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CODE_DIR = Path(os.environ.get("SM_CHANNEL_CODE", "/opt/ml/input/data/code"))

# predict_onnx_csv.py and eer.py sit at the source root, one level above this file.
SOURCE_ROOT = CODE_DIR if CODE_DIR.is_dir() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as smtrain  # noqa: E402

from eer import compute_eer, get_fr_fa_at_threshold  # noqa: E402


def eer_metrics(labels: list[int], scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float32)
    tar = scores_array[labels_array == 1]
    imp = scores_array[labels_array == 0]
    tar = tar[~np.isnan(tar)]
    imp = imp[~np.isnan(imp)]
    if len(tar) == 0 or len(imp) == 0:
        return {
            "bpcer": float("nan"),
            "apcer": float("nan"),
            "acer": float("nan"),
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "threshold": threshold,
        }
    bpcer, apcer = get_fr_fa_at_threshold(tar=tar, imp=imp, threshold=threshold)
    eer, eer_threshold = compute_eer(tar=tar, imp=imp)
    return {
        "bpcer": float(bpcer),
        "apcer": float(apcer),
        "acer": float((bpcer + apcer) / 2.0),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "threshold": threshold,
    }


def read_scores(path: Path) -> tuple[list[int], np.ndarray]:
    labels = []
    scores = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    return labels, np.asarray(scores, dtype=np.float32)


def resolve_device(hps: dict[str, str]) -> str:
    requested = (smtrain.optional(hps, "device") or "auto").lower()
    available_gpus = int(os.environ.get("SM_NUM_GPUS") or 0)
    if requested == "auto":
        return "cuda" if available_gpus > 0 else "cpu"
    if requested == "cuda" and available_gpus == 0:
        raise RuntimeError(
            "device=cuda was requested but this instance has no GPU. "
            "Use a g4dn/g5 instance or pass --device cpu."
        )
    return requested


def copy_converted_artifacts(source_dir: Path, onnx_path: Path) -> dict[str, str]:
    """Copy converted ONNX + JSON into SageMaker output and model artifacts."""
    package_name = source_dir.name
    output_package = OUTPUT_DIR / "converted" / package_name
    model_package = MODEL_DIR / "converted" / package_name
    output_package.parent.mkdir(parents=True, exist_ok=True)
    model_package.parent.mkdir(parents=True, exist_ok=True)
    if output_package.resolve() != source_dir.resolve():
        if output_package.exists():
            shutil.rmtree(output_package)
        shutil.copytree(source_dir, output_package)
    if model_package.exists():
        shutil.rmtree(model_package)
    shutil.copytree(source_dir, model_package)

    flat_onnx = OUTPUT_DIR / onnx_path.name
    shutil.copy2(onnx_path, flat_onnx)
    json_matches = list(source_dir.glob("*.json"))
    flat_json = None
    if json_matches:
        flat_json = OUTPUT_DIR / json_matches[0].name
        shutil.copy2(json_matches[0], flat_json)
        shutil.copy2(json_matches[0], MODEL_DIR / json_matches[0].name)
    shutil.copy2(onnx_path, MODEL_DIR / onnx_path.name)

    print(f"Saved converted ONNX to {flat_onnx} and {model_package / onnx_path.name}")
    return {
        "onnx": str(flat_onnx),
        "onnx_package": str(output_package / onnx_path.name),
        "config": str(flat_json) if flat_json else "",
    }


def keras_files_in_dir(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".keras"
    )


def resolve_checkpoint_files(hps: dict[str, str]) -> list[Path]:
    ckpt_dir = SOURCE_ROOT / "checkpoints"
    raw = smtrain.optional(hps, "checkpoint")
    if raw in (None, "", "*", "all"):
        found = keras_files_in_dir(ckpt_dir)
        if found:
            print(f"Found {len(found)} .keras file(s) in {ckpt_dir} (non-recursive)")
        return found
    files: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_dir():
            found = keras_files_in_dir(path)
            if not found:
                raise FileNotFoundError(f"No .keras files in {path} (non-recursive)")
            files.extend(found)
            continue
        checkpoint = ckpt_dir / Path(item).name
        if not checkpoint.is_file():
            checkpoint = SOURCE_ROOT / Path(item).name
        if not checkpoint.is_file() and path.is_file():
            checkpoint = path
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {item}")
        files.append(checkpoint)
    return files


def load_checkpoint_sources() -> dict[str, dict[str, str]]:
    path = SOURCE_ROOT / "checkpoints" / "sources.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list):
        return {}
    by_name: dict[str, dict[str, str]] = {}
    for item in files:
        if isinstance(item, dict) and item.get("name"):
            by_name[str(item["name"])] = item
    return by_name


def convert_checkpoints(hps: dict[str, str]) -> list[tuple[Path, dict[str, str]]]:
    checkpoints = resolve_checkpoint_files(hps)
    if not checkpoints:
        return []
    converter = SOURCE_ROOT / "convert_checkpoint_to_onnx.py"
    if not converter.is_file():
        raise FileNotFoundError("convert_checkpoint_to_onnx.py was not staged with the job")
    opset = smtrain.optional(hps, "opset") or "17"
    sources = load_checkpoint_sources()
    converted: list[tuple[Path, dict[str, str]]] = []
    used_stems: set[str] = set()
    for checkpoint in checkpoints:
        stem = checkpoint.stem
        if stem in used_stems:
            stem = f"{stem}_{len(used_stems)}"
        used_stems.add(stem)
        work_dir = Path("/tmp/converted") / stem
        if work_dir.exists():
            shutil.rmtree(work_dir)
        command = [
            sys.executable,
            "-u",
            str(converter),
            str(checkpoint),
            "--output-dir",
            str(work_dir),
            "--opset",
            opset,
            "--force",
        ]
        print("Converting checkpoint:", " ".join(command))
        subprocess.run(command, check=True)
        onnx_path = work_dir / f"{stem}.onnx"
        if not onnx_path.is_file():
            matches = list(work_dir.glob("*.onnx"))
            if not matches:
                raise FileNotFoundError(f"Conversion produced no ONNX in {work_dir}")
            onnx_path = matches[0]
        artifacts = copy_converted_artifacts(work_dir, onnx_path)
        source = sources.get(checkpoint.name, {})
        artifacts["checkpoint"] = source.get("name") or checkpoint.name
        artifacts["checkpoint_path"] = source.get("path") or str(checkpoint)
        artifacts["checkpoint_arg"] = source.get("checkpoint_arg") or ""
        converted.append((Path(artifacts["onnx"]), artifacts))
    return converted


def onnx_files(hps: dict[str, str]) -> list[Path]:
    raw = smtrain.optional(hps, "onnx")
    search_roots = [CODE_DIR / "models", CODE_DIR]
    if raw:
        candidates = []
        for item in raw.split(","):
            item = item.strip()
            path = Path(item)
            if not path.is_file():
                path = CODE_DIR / item
            if not path.is_file():
                path = CODE_DIR / "models" / Path(item).name
            if path.is_file():
                candidates.append(path)
            else:
                raise FileNotFoundError(f"ONNX model not found: {item}")
        return candidates
    found = []
    for root in search_roots:
        if root.is_dir():
            found.extend(sorted(root.glob("*.onnx")))
            found.extend(sorted(root.glob("**/*.onnx")))
    unique = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(SOURCE_ROOT)

    hps = smtrain.load_hyperparameters()
    input_csv = smtrain.resolve_file(smtrain.optional(hps, "csv") or "manifests/eval.csv")
    if input_csv is None:
        raise FileNotFoundError("Eval CSV not found")
    rewritten = smtrain.rewrite_csv(input_csv, Path("/tmp/sagemaker_csvs/eval.csv"), smtrain.prefix_maps(hps))

    batch_size = smtrain.optional(hps, "batch_size") or "32"
    device = resolve_device(hps)
    print(f"onnxruntime device: {device} (SM_NUM_GPUS={os.environ.get('SM_NUM_GPUS')})")
    threshold = float(smtrain.optional(hps, "eer_threshold") or "0.5")
    use_bbox = smtrain.truthy(hps.get("use_bbox_crop", "false"))
    margin = smtrain.optional(hps, "margin") or "0.0"
    on_missing = smtrain.optional(hps, "on_missing") or "skip"

    predict_script = SOURCE_ROOT / "predict_onnx_csv.py"
    reports = []
    converted_models = convert_checkpoints(hps)
    models: list[tuple[Path, dict[str, str] | None]] = [(path, artifacts) for path, artifacts in converted_models]
    models.extend((path, None) for path in onnx_files(hps))
    if not models:
        raise FileNotFoundError("No models to evaluate. Pass --onnx and/or --checkpoint.")
    for model_path, artifacts in models:
        stem = model_path.stem
        pred_csv = OUTPUT_DIR / f"{stem}_predictions.csv"
        command = [
            sys.executable, "-u", str(predict_script),
            str(rewritten),
            "--model", str(model_path),
            "--output-csv", str(pred_csv),
            "--batch-size", str(batch_size),
            "--device", device,
            "--on-missing", on_missing,
            "--margin", str(margin),
        ]
        if use_bbox:
            command.append("--use-bbox-crop")
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
        labels, scores = read_scores(pred_csv)
        metrics = eer_metrics(labels, scores, threshold)
        entry = {
            "model": str(model_path.name),
            "predictions": str(pred_csv),
            "samples": len(labels),
            "metrics": metrics,
        }
        if artifacts:
            entry["converted_onnx"] = artifacts["onnx"]
            entry["converted_package"] = artifacts["onnx_package"]
            if artifacts.get("config"):
                entry["converted_config"] = artifacts["config"]
            if artifacts.get("checkpoint"):
                entry["checkpoint"] = artifacts["checkpoint"]
            if artifacts.get("checkpoint_path"):
                entry["checkpoint_path"] = artifacts["checkpoint_path"]
            if artifacts.get("checkpoint_arg"):
                entry["checkpoint_arg"] = artifacts["checkpoint_arg"]
        reports.append(entry)
        print(json.dumps(reports[-1], indent=2))

    report_path = OUTPUT_DIR / "eval_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (MODEL_DIR / "eval_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
