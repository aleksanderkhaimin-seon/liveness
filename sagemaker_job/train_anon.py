#!/usr/bin/env python3
"""SageMaker Training Job entrypoint for anonymised-patch training.

Reads the original (rewritten) train CSV from EFS, samples texture patches
with sample_document_patches.py onto local disk, drops the re-identification
manifest, then runs train_efficientnet_b2.py on patches.csv.

Validation and test CSVs are sampled into patches as well (anonymize_eval,
default true from launch_anon.py); they must have a bbox column. Scoring a
patch-trained model on full frames measures a 4-8x scale mismatch rather
than the model, so anonymize_eval=false is only for demonstrating that.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(os.environ.get("SM_CHANNEL_CODE", "/opt/ml/input/data/code"))
sys.path.insert(0, str(CODE_DIR if CODE_DIR.is_dir() else Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as smtrain  # noqa: E402

CODE_DIR = smtrain.CODE_DIR
SM_MODEL_DIR = smtrain.SM_MODEL_DIR
PATCH_ROOT = Path(os.environ.get("ANON_PATCH_DIR", "/tmp/anon_patches"))

SAMPLER_DEFAULTS = {
    "patch_size": "512",
    "patches_per_image": "8",
    "patch_margin": "25",
    "patch_zone_interior": "0.5",
    "patch_zone_edge": "0.5",
    "patch_zone_exterior": "0.0",
    "patch_edge_jitter": "0.2",
    "patch_min_zone_frac": "0.15",
    "patch_min_center_distance": "0.25",
    "patch_max_tries": "40",
    "patch_max_exclusion_overlap": "0.0",
    "patch_min_std": "0.0",
    "patch_format": "png",
    "patch_quality": "100",
    "patch_seed": "0",
    "patch_workers": "4",
}


def hp(hps: dict[str, str], key: str) -> str:
    value = smtrain.optional(hps, key)
    if value is not None:
        return value
    return SAMPLER_DEFAULTS[key]


def staged_script(name: str) -> Path:
    script = CODE_DIR / name
    if not script.is_file():
        script = Path(__file__).resolve().parents[1] / name
    if not script.is_file():
        raise FileNotFoundError(f"{name} was not found in the job source")
    return script


def sampler_script() -> Path:
    return staged_script("sample_document_patches.py")


def drop_manifest(out_dir: Path) -> None:
    manifest = out_dir / "manifest.csv"
    if manifest.is_file():
        manifest.unlink()
        print(f"Removed re-identification map {manifest}")


def aggregate_test_patches(manifest: Path, split: str = "test") -> None:
    """Per-zone and per-document EER for a patch-sampled eval split.

    Runs while the manifest still exists, writes only aggregates (no source
    paths) into the model dir, and folds them into report.json under
    patch_eer.<split> so the MLflow logger picks them up as metrics.
    """
    predictions = SM_MODEL_DIR / f"{split}_predictions.csv"
    if not predictions.is_file():
        print(f"No {predictions.name}; skipping patch aggregation for {split}")
        return
    json_out = SM_MODEL_DIR / f"patch_eer_{split}.json"
    text_out = SM_MODEL_DIR / f"patch_eer_{split}.txt"
    command = [
        sys.executable,
        "-u",
        str(staged_script("aggregate_patch_eer.py")),
        str(predictions),
        str(manifest),
        "--json-out",
        str(json_out),
    ]
    print(f"Aggregating {split} patch predictions per zone and per document:")
    print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    text_out.write_text(result.stdout + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    print(result.stdout)
    if result.returncode != 0:
        print(f"Patch aggregation failed (exit {result.returncode}); see {text_out}")
        return

    report_path = SM_MODEL_DIR / "report.json"
    if report_path.is_file() and json_out.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.setdefault("patch_eer", {})[split] = json.loads(json_out.read_text(encoding="utf-8"))
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Added patch_eer.{split} to {report_path}")


def zone_weights(hps: dict[str, str]) -> str:
    return (
        f"interior={hp(hps, 'patch_zone_interior')},"
        f"edge={hp(hps, 'patch_zone_edge')},"
        f"exterior={hp(hps, 'patch_zone_exterior')}"
    )


def sample_split(name: str, input_csv: Path, hps: dict[str, str], keep_manifest: bool = False) -> Path:
    out_dir = PATCH_ROOT / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        str(sampler_script()),
        str(input_csv),
        "--out-dir",
        str(out_dir),
        "--patch-size",
        hp(hps, "patch_size"),
        "--patches-per-image",
        hp(hps, "patches_per_image"),
        "--margin",
        hp(hps, "patch_margin"),
        "--zone-weights",
        zone_weights(hps),
        "--edge-jitter",
        hp(hps, "patch_edge_jitter"),
        "--min-zone-frac",
        hp(hps, "patch_min_zone_frac"),
        "--min-center-distance",
        hp(hps, "patch_min_center_distance"),
        "--max-tries",
        hp(hps, "patch_max_tries"),
        "--max-exclusion-overlap",
        hp(hps, "patch_max_exclusion_overlap"),
        "--min-std",
        hp(hps, "patch_min_std"),
        "--format",
        hp(hps, "patch_format"),
        "--quality",
        hp(hps, "patch_quality"),
        "--seed",
        hp(hps, "patch_seed"),
        "--workers",
        hp(hps, "patch_workers"),
        "--fail-fast",
    ]
    if smtrain.truthy(hps.get("patch_no_backfill")):
        command.append("--no-backfill")
    if smtrain.truthy(hps.get("patch_clip_to_margin")):
        command.append("--clip-to-margin")
    exclude_column = smtrain.optional(hps, "patch_exclude_column")
    if exclude_column:
        command.extend(["--exclude-column", exclude_column])
    limit = smtrain.optional(hps, "patch_limit")
    if limit and int(limit) > 0:
        command.extend(["--limit", limit])

    print(f"Sampling {name} patches:")
    print(" ".join(command))
    subprocess.run(command, check=True)

    patches_csv = out_dir / "patches.csv"
    if not patches_csv.is_file():
        raise FileNotFoundError(f"Sampler did not write {patches_csv}")

    summary_src = out_dir / "summary.json"
    if summary_src.is_file():
        dest = SM_MODEL_DIR / f"patch_summary_{name}.json"
        shutil.copy2(summary_src, dest)
        print(f"Saved {dest}")

    if name != "train":
        # Opaque patch id, label, zone -- no source paths -- so per-zone numbers
        # stay recoverable after the manifest is gone.
        shutil.copy2(patches_csv, SM_MODEL_DIR / f"patch_index_{name}.csv")

    if keep_manifest:
        print(f"Keeping {out_dir / 'manifest.csv'} until after aggregation")
    else:
        drop_manifest(out_dir)

    return patches_csv


def resolve_config_csv(config_path: Path | None, key: str, hps: dict[str, str]) -> Path | None:
    if config_path is not None:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        value = data.get(key)
        if value not in (None, ""):
            return smtrain.resolve_file(str(value))
    return smtrain.resolve_file(smtrain.optional(hps, key)) if smtrain.optional(hps, key) else None


def replace_flag(command: list[str], flag: str, value: str | None) -> list[str]:
    """Drop any existing occurrence of `flag` [value] and append the new pair."""
    cleaned: list[str] = []
    skip_next = False
    for item in command:
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            skip_next = value is not None
            continue
        cleaned.append(item)
    if value is not None:
        cleaned.extend([flag, value])
    else:
        cleaned.append(flag)
    return cleaned


def build_command(hps: dict[str, str], train_csv: Path, val_csv: Path | None, test_csv: Path | None) -> list[str]:
    train_hps = dict(hps)
    train_hps["use_bbox_crop"] = "false"
    train_hps["bbox_aug_prob"] = "0"
    command = smtrain.build_command(train_hps)
    command = replace_flag(command, "--csv", str(train_csv))
    if val_csv is not None:
        command = replace_flag(command, "--validation-csv", str(val_csv))
    if test_csv is not None:
        command = replace_flag(command, "--test-csv", str(test_csv))
    command = [item for item in command if item != "--use-bbox-crop"]
    if "--no-use-bbox-crop" not in command:
        command.append("--no-use-bbox-crop")
    command = replace_flag(command, "--bbox-aug-prob", "0")
    return command


def main() -> None:
    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(CODE_DIR if CODE_DIR.is_dir() else Path.cwd())
    sys.path.insert(0, str(CODE_DIR if CODE_DIR.is_dir() else Path.cwd()))

    hps = smtrain.load_hyperparameters()
    maps = smtrain.prefix_maps(hps)
    config_path = smtrain.staged_config(hps)
    if config_path is not None:
        smtrain.rewrite_config_manifests(config_path, maps)

    train_source = resolve_config_csv(config_path, "csv", hps)
    if train_source is None:
        raise FileNotFoundError("Training CSV not found. Set csv in the staged config.")

    print("SageMaker channels:")
    for name in sorted(path.name for path in smtrain.SM_INPUT_DATA_DIR.iterdir() if path.is_dir()):
        if name in smtrain.RESERVED_CHANNELS:
            continue
        print(f"  {name}: {smtrain.SM_INPUT_DATA_DIR / name}")

    train_csv = sample_split("train", train_source, hps)
    val_csv = resolve_config_csv(config_path, "validation_csv", hps)
    test_csv = resolve_config_csv(config_path, "test_csv", hps)
    test_manifest: Path | None = None
    if smtrain.truthy(hps.get("anonymize_eval")):
        if val_csv is not None:
            val_csv = sample_split("validation", val_csv, hps)
        if test_csv is not None:
            test_csv = sample_split("test", test_csv, hps, keep_manifest=True)
            test_manifest = test_csv.parent / "manifest.csv"
    else:
        print(
            "WARNING: anonymize_eval=false -- validation/test are FULL FRAMES while training "
            "is native-resolution patches. EER from this job reflects the scale mismatch.",
            flush=True,
        )

    command = build_command(hps, train_csv, val_csv, test_csv)
    print("Launching:")
    print(" ".join(command))
    try:
        subprocess.run(command, check=True)
        if test_manifest is not None and test_manifest.is_file():
            aggregate_test_patches(test_manifest, "test")
    finally:
        # Whatever happened above, the re-identification map does not leave the job.
        if test_manifest is not None:
            drop_manifest(test_manifest.parent)

    if config_path is not None:
        shutil.copy2(config_path, SM_MODEL_DIR / config_path.name)
        shutil.copy2(config_path, SM_MODEL_DIR / "train_config.json")
        print(f"Saved original config to {SM_MODEL_DIR / config_path.name}")
    try:
        smtrain.log_mlflow_run(hps)
    except Exception as error:
        print(f"MLflow logging failed (training artifacts are still saved): {error}")


if __name__ == "__main__":
    main()
