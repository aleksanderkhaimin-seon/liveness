#!/usr/bin/env python3
"""Submit ONNX evaluation as a SageMaker Training Job (Jobs tab).

SageMaker has no separate ONNX eval job type. This reuses the same custom
image and EFS mounts as training, with sagemaker_job/evaluate.py as the
entrypoint. Results land in the job output S3 prefix as eval_report.json
and <model>_predictions.csv.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes.shapes import FileSystemDataSource, OutputDataConfig, StoppingCondition
from sagemaker.core.training.configs import Compute, InputData, Networking, SourceCode
from sagemaker.train import ModelTrainer

import launch as train_launch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CSV = REPO_ROOT / "data" / "ProdTest-0.2-val.csv"


def collect_onnx(paths: list[Path] | None) -> list[Path]:
    if not paths:
        return []
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.onnx")))
            files.extend(sorted(path.glob("**/*.onnx")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"--onnx not found: {path}")
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved.suffix.lower() != ".onnx":
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if paths and not unique:
        raise FileNotFoundError("No .onnx files found under --onnx.")
    return unique


def keras_files_in_dir(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".keras"
    )


def collect_checkpoints(paths: list[Path] | None) -> tuple[list[Path], list[dict[str, str]]]:
    if not paths:
        return [], []
    files: list[Path] = []
    records: list[dict[str, str]] = []
    for path in paths:
        if path.is_dir():
            found = keras_files_in_dir(path)
            if not found:
                raise FileNotFoundError(f"No .keras files in {path} (non-recursive)")
            print(f"Found {len(found)} .keras file(s) in {path} (non-recursive)")
            for keras_path in found:
                print(f"  {keras_path.name}")
                files.append(keras_path)
                records.append(
                    {
                        "name": keras_path.name,
                        "path": str(keras_path.resolve()),
                        "checkpoint_arg": str(path),
                    }
                )
        elif path.is_file():
            files.append(path)
            records.append(
                {
                    "name": path.name,
                    "path": str(path.resolve()),
                    "checkpoint_arg": str(path),
                }
            )
        else:
            raise FileNotFoundError(f"--checkpoint not found: {path}")
    unique = []
    unique_records = []
    seen = set()
    for path, record in zip(files, records):
        resolved = path.resolve()
        if resolved.suffix.lower() != ".keras":
            raise ValueError(f"--checkpoint must be a .keras file or directory, got {path}")
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
            unique_records.append(record)
    if paths and not unique:
        raise FileNotFoundError("No .keras files found under --checkpoint.")
    return unique, unique_records


def stage_source(
    eval_csv: Path,
    onnx_files: list[Path],
    checkpoints: list[Path],
    checkpoint_records: list[dict[str, str]],
    checkpoint_args: list[Path] | None,
) -> Path:
    stage = Path(tempfile.mkdtemp(prefix="liveness-sagemaker-eval-"))
    shutil.copy2(REPO_ROOT / "predict_onnx_csv.py", stage / "predict_onnx_csv.py")
    shutil.copy2(REPO_ROOT / "eer.py", stage / "eer.py")
    shutil.copy2(REPO_ROOT / "convert_checkpoint_to_onnx.py", stage / "convert_checkpoint_to_onnx.py")
    shutil.copy2(REPO_ROOT / "train_efficientnet_b2.py", stage / "train_efficientnet_b2.py")
    job_dir = stage / "sagemaker_job"
    job_dir.mkdir()
    shutil.copy2(REPO_ROOT / "sagemaker_job" / "evaluate.py", job_dir / "evaluate.py")
    shutil.copy2(REPO_ROOT / "sagemaker_job" / "train.py", job_dir / "train.py")
    manifests = stage / "manifests"
    manifests.mkdir()
    shutil.copy2(eval_csv, manifests / "eval.csv")
    models = stage / "models"
    models.mkdir()
    for onnx_path in onnx_files:
        shutil.copy2(onnx_path, models / onnx_path.name)
    ckpt_dir = stage / "checkpoints"
    ckpt_dir.mkdir()
    for checkpoint in checkpoints:
        shutil.copy2(checkpoint, ckpt_dir / checkpoint.name)
    if checkpoint_records:
        sources = {
            "checkpoint_args": [str(path) for path in (checkpoint_args or [])],
            "files": checkpoint_records,
        }
        (ckpt_dir / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    return stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ONNX evaluation as a SageMaker Job.")
    parser.add_argument("--image-uri", default=train_launch.DEFAULT_IMAGE)
    parser.add_argument("--role", default=None)
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--volume-size", type=int, default=50)
    parser.add_argument("--max-run", type=int, default=21600)
    parser.add_argument("--base-job-name", default="liveness-onnx-eval")
    parser.add_argument("--output-s3", default=None)
    parser.add_argument("--csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--onnx", type=Path, nargs="*", default=None, help="ONNX file(s) or directories.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="*",
        default=None,
        help="Optional .keras file(s), or a directory of .keras files (non-recursive).",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eer-threshold", type=float, default=0.5)
    parser.add_argument("--use-bbox-crop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--on-missing", choices=["skip", "raise"], default="skip")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto picks cuda only when the instance has a GPU.",
    )
    parser.add_argument("--efs-id", default=train_launch.DEFAULT_EFS_ID)
    parser.add_argument("--datalake-host-path", default=train_launch.DEFAULT_DATALAKE_HOST)
    parser.add_argument("--prod-host-path", default=train_launch.DEFAULT_PROD_HOST)
    parser.add_argument("--subnets", nargs="*", default=None)
    parser.add_argument("--security-group-ids", nargs="*", default=None)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"--csv not found: {args.csv}")
    onnx_files = collect_onnx(args.onnx)
    checkpoints, checkpoint_records = collect_checkpoints(args.checkpoint)
    if not onnx_files and not checkpoints:
        raise FileNotFoundError("Pass --onnx and/or --checkpoint.")

    session = Session()
    role = args.role or get_execution_role()
    efs_directory = train_launch.efs_directory_path(args.datalake_host_path, args.efs_id)
    prefixes = train_launch.strip_prefix_aliases(efs_directory, args.efs_id)
    prod_directory = train_launch.efs_directory_path(args.prod_host_path, args.efs_id)
    prod_prefixes = train_launch.strip_prefix_aliases(prod_directory, args.efs_id)
    prod_host = str(Path(args.prod_host_path)).rstrip("/")
    if prod_host not in prod_prefixes:
        prod_prefixes.append(prod_host)
    train_launch.check_manifest_paths(args.csv, prefixes + prod_prefixes)

    detected_subnets, detected_sgs = train_launch.detect_networking()
    subnets = args.subnets or detected_subnets
    security_groups = args.security_group_ids or detected_sgs
    if not subnets or not security_groups:
        raise RuntimeError("Pass --subnets and --security-group-ids so the job can mount EFS.")

    hyperparameters = {
        "csv": "manifests/eval.csv",
        "batch_size": str(args.batch_size),
        "eer_threshold": str(args.eer_threshold),
        "use_bbox_crop": "true" if args.use_bbox_crop else "false",
        "margin": str(args.margin),
        "on_missing": args.on_missing,
        "device": args.device,
        "strip_path_prefix": "|".join(prefixes),
        "prod_strip_path_prefix": "|".join(prod_prefixes),
    }
    if onnx_files:
        hyperparameters["onnx"] = ",".join(path.name for path in onnx_files)
    if checkpoints:
        joined = ",".join(path.name for path in checkpoints)
        hyperparameters["checkpoint"] = joined if len(joined) < 240 else "*"
        hyperparameters["opset"] = str(args.opset)

    stage = stage_source(args.csv, onnx_files, checkpoints, checkpoint_records, args.checkpoint)
    try:
        trainer = ModelTrainer(
            training_image=args.image_uri,
            role=role,
            sagemaker_session=session,
            base_job_name=args.base_job_name,
            source_code=SourceCode(
                source_dir=str(stage),
                entry_script="sagemaker_job/evaluate.py",
            ),
            compute=Compute(
                instance_type=args.instance_type,
                instance_count=args.instance_count,
                volume_size_in_gb=args.volume_size,
            ),
            stopping_condition=StoppingCondition(max_runtime_in_seconds=args.max_run),
            networking=Networking(subnets=subnets, security_group_ids=security_groups),
            output_data_config=OutputDataConfig(s3_output_path=args.output_s3) if args.output_s3 else None,
            hyperparameters=hyperparameters,
            environment={"PYTHONUNBUFFERED": "1"},
            input_data_config=[
                InputData(
                    channel_name="datalake",
                    data_source=FileSystemDataSource(
                        file_system_id=args.efs_id,
                        file_system_type="EFS",
                        directory_path=efs_directory,
                        file_system_access_mode="ro",
                    ),
                ),
                InputData(
                    channel_name="prod",
                    data_source=FileSystemDataSource(
                        file_system_id=args.efs_id,
                        file_system_type="EFS",
                        directory_path=prod_directory,
                        file_system_access_mode="ro",
                    ),
                ),
            ],
        )
        print("Submitting SageMaker evaluation job")
        print(f"  csv: {args.csv}")
        print(f"  onnx: {[path.name for path in onnx_files]}")
        print(f"  checkpoint: {[path.name for path in checkpoints]}")
        print(f"  instance: {args.instance_type}")
        print(f"  device: {args.device}")
        trainer.train(wait=args.wait, logs=args.wait)
        job = trainer._latest_training_job
        if job is not None:
            name = getattr(job, "training_job_name", None) or getattr(job, "name", None)
            print(f"Evaluation job: {name}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
