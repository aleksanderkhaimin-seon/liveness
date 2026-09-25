#!/usr/bin/env python3
"""SageMaker Training Job entrypoint.

Maps SageMaker channels, hyperparameters, and output paths onto
train_efficientnet_b2.py without changing that script.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
SM_INPUT_DATA_DIR = Path(os.environ.get("SM_INPUT_DATA_DIR", "/opt/ml/input/data"))
SM_HPS_PATH = Path("/opt/ml/input/config/hyperparameters.json")
CODE_DIR = Path(os.environ.get("SM_CHANNEL_CODE", "/opt/ml/input/data/code"))
RESERVED_CHANNELS = {"code", "sm_drivers", "recipe"}
DEFAULT_STRIP_PREFIXES = [
    "/home/sagemaker-user/seon-data-efs/data",
    "/mnt/custom-file-systems/efs/fs-0773949cd1f9915ec/seon-data-efs/data",
    "/home/sagemaker-user/custom-file-systems/efs/fs-0773949cd1f9915ec/seon-data-efs/data",
    "/mnt/dataefs/data",
]
DEFAULT_PROD_STRIP_PREFIXES = [
    "/home/sagemaker-user/prod-data-efs/buckets",
    "/mnt/custom-file-systems/efs/fs-0773949cd1f9915ec/prod-data-efs/buckets",
    "/home/sagemaker-user/custom-file-systems/efs/fs-0773949cd1f9915ec/prod-data-efs/buckets",
]


def load_hyperparameters() -> dict[str, str]:
    if not SM_HPS_PATH.exists():
        return {}
    raw = json.loads(SM_HPS_PATH.read_text(encoding="utf-8"))
    return {str(key).replace("-", "_"): "" if value is None else str(value) for key, value in raw.items()}


def truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def optional(hps: dict[str, str], key: str) -> str | None:
    value = hps.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def channel_dir(name: str) -> Path | None:
    env_value = os.environ.get(f"SM_CHANNEL_{name.upper()}")
    if env_value:
        path = Path(env_value)
        if path.is_dir():
            return path
    path = SM_INPUT_DATA_DIR / name
    return path if path.is_dir() else None


def parse_prefixes(hps: dict[str, str], key: str, defaults: list[str]) -> list[str]:
    raw = optional(hps, key)
    if not raw:
        return defaults
    prefixes = [item.strip().rstrip("/") for item in raw.split("|") if item.strip()]
    return prefixes or defaults


def prefix_maps(hps: dict[str, str]) -> list[tuple[list[str], Path]]:
    maps: list[tuple[list[str], Path]] = []
    datalake_dir = channel_dir("datalake")
    image_root = Path(optional(hps, "image_root")) if optional(hps, "image_root") else datalake_dir
    if image_root is None:
        raise FileNotFoundError("EFS datalake channel was not mounted at /opt/ml/input/data/datalake")
    maps.append((parse_prefixes(hps, "strip_path_prefix", DEFAULT_STRIP_PREFIXES), image_root))

    prod_dir = channel_dir("prod")
    if prod_dir is not None:
        maps.append((parse_prefixes(hps, "prod_strip_path_prefix", DEFAULT_PROD_STRIP_PREFIXES), prod_dir))
    return maps


def resolve_file(preferred: str | None) -> Path | None:
    if not preferred:
        return None
    path = Path(preferred)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                CODE_DIR / path,
                CODE_DIR / "manifests" / path.name,
                Path.cwd() / path,
            ]
        )
    else:
        candidates.append(CODE_DIR / "data" / path.name)
        candidates.append(CODE_DIR / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"File not found: {preferred}")


resolve_csv = resolve_file


def rewrite_csv(source: Path, destination: Path, maps: list[tuple[list[str], Path]]) -> Path:
    rules: list[tuple[str, Path]] = []
    for prefixes, image_root in maps:
        for prefix in prefixes:
            cleaned = prefix.rstrip("/")
            if cleaned:
                rules.append((cleaned, image_root))
    rules.sort(key=lambda item: len(item[0]), reverse=True)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8", newline="") as incoming, destination.open(
        "w", encoding="utf-8", newline=""
    ) as outgoing:
        reader = csv.DictReader(incoming)
        if not reader.fieldnames:
            raise ValueError(f"{source} has no header")
        writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames)
        writer.writeheader()
        rewritten = 0
        for row in reader:
            image_path = (row.get("path") or "").strip()
            for prefix, image_root in rules:
                if image_path == prefix or image_path.startswith(prefix + "/"):
                    remainder = image_path[len(prefix) :].lstrip("/")
                    row["path"] = str(image_root / remainder)
                    rewritten += 1
                    break
            writer.writerow(row)
    print(f"Rewrote {rewritten} image paths in {source.name} -> {destination}")
    return destination


def rewrite_csv_inplace(source: Path, maps: list[tuple[list[str], Path]]) -> Path:
    rewrite_dir = Path("/tmp/sagemaker_csvs")
    rewrite_dir.mkdir(parents=True, exist_ok=True)
    tmp = rewrite_dir / source.name
    if tmp.resolve() == source.resolve():
        tmp = rewrite_dir / f"rewritten_{source.name}"
    rewritten = rewrite_csv(source, tmp, maps)
    if rewritten.resolve() != source.resolve():
        shutil.copy2(rewritten, source)
    return source


def staged_config(hps: dict[str, str]) -> Path | None:
    return resolve_file(optional(hps, "config"))


def rewrite_config_manifests(config_path: Path, maps: list[tuple[list[str], Path]]) -> None:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    for key in ("csv", "validation_csv", "test_csv"):
        value = data.get(key)
        if value in (None, ""):
            continue
        source = resolve_file(str(value))
        rewrite_csv_inplace(source, maps)


def build_command(hps: dict[str, str]) -> list[str]:
    script = CODE_DIR / "train_efficientnet_b2.py"
    if not script.is_file():
        script = Path(__file__).resolve().parents[1] / "train_efficientnet_b2.py"
    if not script.is_file():
        raise FileNotFoundError("train_efficientnet_b2.py was not found in the job source")

    maps = prefix_maps(hps)
    config_path = staged_config(hps)
    if config_path is not None:
        rewrite_config_manifests(config_path, maps)
        command = [
            sys.executable,
            "-u",
            str(script),
            "--config",
            str(config_path),
            "--output-dir",
            str(SM_MODEL_DIR),
        ]
    else:
        train_csv = resolve_file(optional(hps, "csv"))
        if train_csv is None:
            raise FileNotFoundError("Training CSV not found. Set csv in the staged config.")
        rewrite_dir = Path("/tmp/sagemaker_csvs")
        train_csv = rewrite_csv(train_csv, rewrite_dir / "train.csv", maps)
        command = [sys.executable, "-u", str(script), "--csv", str(train_csv), "--output-dir", str(SM_MODEL_DIR)]
        validation_csv = resolve_file(optional(hps, "validation_csv")) if optional(hps, "validation_csv") else None
        if validation_csv is not None:
            command.extend(["--validation-csv", str(rewrite_csv(validation_csv, rewrite_dir / "validation.csv", maps))])
        test_csv = resolve_file(optional(hps, "test_csv")) if optional(hps, "test_csv") else None
        if test_csv is not None:
            command.extend(["--test-csv", str(rewrite_csv(test_csv, rewrite_dir / "test.csv", maps))])

    scalar_flags = {
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "learning_rate": "--learning-rate",
        "min_learning_rate": "--min-learning-rate",
        "validation_split": "--validation-split",
        "seed": "--seed",
        "eer_threshold": "--eer-threshold",
        "margin": "--margin",
        "bbox_aug_prob": "--bbox-aug-prob",
        "degrade": "--degrade",
        "comment": "--comment",
    }
    for key, flag in scalar_flags.items():
        value = optional(hps, key)
        if value is not None and key != "comment":
            command.extend([flag, value])
        elif key == "comment" and value:
            command.extend([flag, value])

    bool_flags = {
        "cosine_decay": ("--cosine-decay", "--no-cosine-decay"),
        "train_backbone": ("--train-backbone", "--no-train-backbone"),
        "require_gpu": ("--require-gpu", "--no-require-gpu"),
        "mixed_precision": ("--mixed-precision", "--no-mixed-precision"),
        "use_bbox_crop": ("--use-bbox-crop", "--no-use-bbox-crop"),
    }
    for key, (on_flag, off_flag) in bool_flags.items():
        if key not in hps:
            continue
        command.append(on_flag if truthy(hps[key]) else off_flag)

    return command


def _as_float(value) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten_metrics(payload, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            metrics.update(flatten_metrics(value, name))
    elif isinstance(payload, list):
        if payload and all(_as_float(item) is not None for item in payload):
            last = _as_float(payload[-1])
            if last is not None:
                metrics[prefix] = last
    else:
        number = _as_float(payload)
        if number is not None and prefix:
            metrics[prefix] = number
    return metrics


def ensure_mlflow():
    try:
        import mlflow
        import sagemaker_mlflow  # noqa: F401
        return mlflow
    except ImportError:
        print("Installing mlflow and sagemaker-mlflow in the job...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "mlflow", "sagemaker-mlflow"],
            check=True,
        )
        import mlflow
        import sagemaker_mlflow  # noqa: F401
        return mlflow


def log_mlflow_run(hps: dict[str, str]) -> None:
    tracking_uri = (
        os.environ.get("MLFLOW_TRACKING_ARN")
        or os.environ.get("MLFLOW_TRACKING_URI")
        or optional(hps, "mlflow_tracking_arn")
    )
    if not tracking_uri:
        print("MLflow skipped: no MLFLOW_TRACKING_ARN on the job")
        return
    mlflow = ensure_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = (
        os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or optional(hps, "mlflow_experiment")
        or "liveness-efficientnet-b2"
    )
    mlflow.set_experiment(experiment)
    run_name = os.environ.get("TRAINING_JOB_NAME") or "liveness"
    skip_params = {"strip_path_prefix", "prod_strip_path_prefix"}
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("sagemaker_job_name", run_name)
        for key, value in hps.items():
            if key in skip_params or value is None:
                continue
            text = str(value)
            if 0 < len(text) <= 250:
                mlflow.log_param(key, text)
        history_path = SM_MODEL_DIR / "history.csv"
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    step = int(float(row["epoch"])) if row.get("epoch") not in (None, "") else None
                    metrics = {}
                    for column, raw in row.items():
                        if column == "epoch":
                            continue
                        number = _as_float(raw)
                        if number is not None:
                            metrics[column] = number
                    if metrics:
                        mlflow.log_metrics(metrics, step=step)
            mlflow.log_artifact(str(history_path))
        report_path = SM_MODEL_DIR / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for name, number in flatten_metrics(report).items():
                if name.startswith("history."):
                    continue
                mlflow.log_metric(name.replace(".", "_")[:250], number)
            mlflow.log_artifact(str(report_path))
        for name in ("train_config.json",):
            path = SM_MODEL_DIR / name
            if path.is_file():
                mlflow.log_artifact(str(path))
        print(f"Logged MLflow run to {tracking_uri} experiment={experiment} run={run_name}")


def main() -> None:
    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(CODE_DIR if CODE_DIR.is_dir() else Path.cwd())
    sys.path.insert(0, str(CODE_DIR if CODE_DIR.is_dir() else Path.cwd()))

    hps = load_hyperparameters()
    command = build_command(hps)
    print("SageMaker channels:")
    for name in sorted(path.name for path in SM_INPUT_DATA_DIR.iterdir() if path.is_dir()):
        if name in RESERVED_CHANNELS:
            continue
        print(f"  {name}: {SM_INPUT_DATA_DIR / name}")
    print("Launching:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    config_path = staged_config(hps)
    if config_path is not None:
        shutil.copy2(config_path, SM_MODEL_DIR / config_path.name)
        shutil.copy2(config_path, SM_MODEL_DIR / "train_config.json")
        print(f"Saved original config to {SM_MODEL_DIR / config_path.name}")
    try:
        log_mlflow_run(hps)
    except Exception as error:
        print(f"MLflow logging failed (training artifacts are still saved): {error}")


if __name__ == "__main__":
    main()
