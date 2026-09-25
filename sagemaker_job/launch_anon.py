#!/usr/bin/env python3
"""Submit anonymised-patch EfficientNetB2 training as a SageMaker Job.

The job samples texture patches from the train CSV with sample_document_patches.py
(local EBS, originals stay on EFS), then trains on patches.csv. The
re-identification manifest is deleted inside the job and is not uploaded.

Example:

    python sagemaker_job/launch_anon.py \\
      --config configs/train-gpu.json \\
      --instance-type ml.g5.xlarge \\
      --volume-size 200
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes.shapes import FileSystemDataSource, OutputDataConfig, StoppingCondition
from sagemaker.core.training.configs import Compute, InputData, Networking, SourceCode
from sagemaker.train import ModelTrainer

import launch as train_launch

REPO_ROOT = Path(__file__).resolve().parents[1]
ZONES = ("interior", "edge", "exterior")
DEFAULT_MLFLOW_EXPERIMENT = "liveness-anon-efficientnet-b2"


def parse_weights(text: str) -> dict[str, float]:
    weights = {zone: 0.0 for zone in ZONES}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in weights:
            raise argparse.ArgumentTypeError(f"Unknown zone {name!r}, expected one of {ZONES}")
        weights[name] = float(value)
    if sum(weights.values()) <= 0:
        raise argparse.ArgumentTypeError("At least one zone weight must be positive")
    return weights


def require_bbox_column(csv_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "bbox" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must contain a bbox column so sample_document_patches.py can run. "
                "Use a *-bbox.csv (or a detector-annotated CSV) for --config csv."
            )


def stage_source(config_path: Path, config: dict, csv_paths: dict[str, Path]) -> tuple[Path, Path]:
    stage, config_rel = train_launch.stage_source(config_path, config, csv_paths)
    shutil.copy2(REPO_ROOT / "sample_document_patches.py", stage / "sample_document_patches.py")
    shutil.copy2(REPO_ROOT / "aggregate_patch_eer.py", stage / "aggregate_patch_eer.py")
    shutil.copy2(REPO_ROOT / "sagemaker_job" / "train_anon.py", stage / "sagemaker_job" / "train_anon.py")
    return stage, config_rel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch anonymised-patch liveness training as a SageMaker Training Job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-uri", default=train_launch.DEFAULT_IMAGE)
    parser.add_argument("--role", default=None)
    parser.add_argument("--instance-type", default="ml.g5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument(
        "--volume-size",
        type=int,
        default=200,
        help="EBS GB for code, checkpoints, and sampled PNG patches.",
    )
    parser.add_argument("--max-run", type=int, default=86400)
    parser.add_argument("--base-job-name", default="liveness-anon-efficientnet-b2")
    parser.add_argument("--output-s3", default=None)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.json")
    parser.add_argument("--efs-id", default=train_launch.DEFAULT_EFS_ID)
    parser.add_argument("--datalake-host-path", default=train_launch.DEFAULT_DATALAKE_HOST)
    parser.add_argument("--prod-host-path", default=train_launch.DEFAULT_PROD_HOST)
    parser.add_argument("--efs-directory-path", default=None)
    parser.add_argument("--strip-path-prefix", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cosine-decay", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--validation-split", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eer-threshold", type=float)
    parser.add_argument("--comment", default=None)
    parser.add_argument("--subnets", nargs="*", default=None)
    parser.add_argument("--security-group-ids", nargs="*", default=None)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlflow-tracking-arn", default=None)
    parser.add_argument("--mlflow-app", default=None)
    parser.add_argument("--mlflow-experiment", default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument(
        "--patch-margin",
        type=float,
        default=25.0,
        help="Sampler margin percent (same meaning as sample_document_patches.py --margin).",
    )
    parser.add_argument("--zone-weights", type=parse_weights, default="interior=0.5,edge=0.5,exterior=0.0")
    parser.add_argument("--edge-jitter", type=float, default=0.2)
    parser.add_argument("--min-zone-frac", type=float, default=0.15)
    parser.add_argument("--min-center-distance", type=float, default=0.25)
    parser.add_argument("--max-tries", type=int, default=40)
    parser.add_argument("--no-backfill", action="store_true")
    parser.add_argument("--exclude-column", default=None)
    parser.add_argument("--max-exclusion-overlap", type=float, default=0.0)
    parser.add_argument("--min-std", type=float, default=0.0)
    parser.add_argument("--clip-to-margin", action="store_true")
    parser.add_argument("--format", dest="image_format", choices=("png", "jpeg"), default="png")
    parser.add_argument("--quality", type=int, default=100)
    parser.add_argument("--patch-seed", type=int, default=0)
    parser.add_argument("--patch-limit", type=int, default=0, help="Sampler row cap (0 = all).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--anonymize-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample patches from validation_csv and test_csv too (they must have bbox). "
        "Default on: a patch-trained model scored on full frames sees a 4-8x scale "
        "mismatch and its EER is not interpretable. --no-anonymize-eval only for "
        "deliberately measuring that mismatch.",
    )
    return parser.parse_args()


def build_hyperparameters(config: dict, args: argparse.Namespace, prefixes: list[str], prod_prefixes: list[str]) -> dict[str, str]:
    weights = args.zone_weights if isinstance(args.zone_weights, dict) else parse_weights(args.zone_weights)
    args.use_bbox_crop = False
    args.bbox_aug_prob = 0.0
    args.margin = None
    hyperparameters = train_launch.build_hyperparameters(config, args, prefixes, prod_prefixes)
    hyperparameters["use_bbox_crop"] = "false"
    hyperparameters["bbox_aug_prob"] = "0"
    hyperparameters["anonymize_eval"] = "true" if args.anonymize_eval else "false"
    hyperparameters["patch_size"] = train_launch.stringify(args.patch_size)
    hyperparameters["patches_per_image"] = train_launch.stringify(args.patches_per_image)
    hyperparameters["patch_margin"] = train_launch.stringify(args.patch_margin)
    hyperparameters["patch_zone_interior"] = train_launch.stringify(weights["interior"])
    hyperparameters["patch_zone_edge"] = train_launch.stringify(weights["edge"])
    hyperparameters["patch_zone_exterior"] = train_launch.stringify(weights["exterior"])
    hyperparameters["patch_edge_jitter"] = train_launch.stringify(args.edge_jitter)
    hyperparameters["patch_min_zone_frac"] = train_launch.stringify(args.min_zone_frac)
    hyperparameters["patch_min_center_distance"] = train_launch.stringify(args.min_center_distance)
    hyperparameters["patch_max_tries"] = train_launch.stringify(args.max_tries)
    hyperparameters["patch_no_backfill"] = "true" if args.no_backfill else "false"
    hyperparameters["patch_max_exclusion_overlap"] = train_launch.stringify(args.max_exclusion_overlap)
    hyperparameters["patch_min_std"] = train_launch.stringify(args.min_std)
    hyperparameters["patch_clip_to_margin"] = "true" if args.clip_to_margin else "false"
    hyperparameters["patch_format"] = args.image_format
    hyperparameters["patch_quality"] = train_launch.stringify(args.quality)
    hyperparameters["patch_seed"] = train_launch.stringify(args.patch_seed)
    hyperparameters["patch_workers"] = train_launch.stringify(args.workers)
    if args.exclude_column:
        hyperparameters["patch_exclude_column"] = train_launch.stringify(args.exclude_column)
    if args.patch_limit:
        hyperparameters["patch_limit"] = train_launch.stringify(args.patch_limit)
    return hyperparameters


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"--config not found: {args.config}")
    config = train_launch.load_train_config(args.config)
    csv_paths = train_launch.config_csv_paths(config)
    args.csv = csv_paths["csv"]
    args.validation_csv = csv_paths.get("validation_csv")
    args.test_csv = csv_paths.get("test_csv")
    require_bbox_column(args.csv)
    if args.anonymize_eval:
        for key in ("validation_csv", "test_csv"):
            path = csv_paths.get(key)
            if path is not None:
                require_bbox_column(path)
    else:
        print(
            "WARNING: --no-anonymize-eval: validation and test will be scored on FULL FRAMES "
            "with a model trained on native-resolution patches. The resulting EER measures the "
            "scale mismatch, not the model. runs/patch_expere_1 is what that looks like.",
            flush=True,
        )

    session = Session()
    role = args.role or get_execution_role()
    efs_directory = args.efs_directory_path or train_launch.efs_directory_path(args.datalake_host_path, args.efs_id)
    prefixes = (
        args.strip_path_prefix.split("|")
        if args.strip_path_prefix
        else train_launch.strip_prefix_aliases(efs_directory, args.efs_id)
    )
    prod_directory = train_launch.efs_directory_path(args.prod_host_path, args.efs_id)
    prod_prefixes = train_launch.strip_prefix_aliases(prod_directory, args.efs_id)
    prod_host = str(Path(args.prod_host_path))
    if prod_host.rstrip("/") not in prod_prefixes:
        prod_prefixes.append(prod_host.rstrip("/"))
    combined_prefixes = prefixes + prod_prefixes
    for manifest in (args.csv, args.validation_csv, args.test_csv):
        if manifest is not None:
            train_launch.check_manifest_paths(manifest, combined_prefixes)

    detected_subnets, detected_sgs = train_launch.detect_networking()
    subnets = args.subnets or detected_subnets
    security_groups = args.security_group_ids or detected_sgs
    if not subnets or not security_groups:
        raise RuntimeError(
            "SageMaker must run in the EFS VPC. Pass --subnets and --security-group-ids "
            "(include the Studio default SG and the outbound-NFS SG)."
        )

    mlflow_arn = None
    if args.mlflow:
        mlflow_arn = train_launch.detect_mlflow_tracking_arn(args.mlflow_tracking_arn, args.mlflow_app)
        if not mlflow_arn:
            print(
                "MLflow: no app selected. Pass --mlflow-app ah-exp or --mlflow-tracking-arn. "
                "Training will still submit."
            )

    stage, args.config_rel = stage_source(args.config, config, csv_paths)
    try:
        environment = {"PYTHONUNBUFFERED": "1"}
        if mlflow_arn:
            environment["MLFLOW_TRACKING_ARN"] = mlflow_arn
            environment["MLFLOW_TRACKING_URI"] = mlflow_arn
            environment["MLFLOW_EXPERIMENT_NAME"] = train_launch.stringify(args.mlflow_experiment)
        weights = args.zone_weights if isinstance(args.zone_weights, dict) else parse_weights(args.zone_weights)
        trainer = ModelTrainer(
            training_image=args.image_uri,
            role=role,
            sagemaker_session=session,
            base_job_name=args.base_job_name,
            source_code=SourceCode(
                source_dir=str(stage),
                entry_script="sagemaker_job/train_anon.py",
            ),
            compute=Compute(
                instance_type=args.instance_type,
                instance_count=args.instance_count,
                volume_size_in_gb=args.volume_size,
            ),
            stopping_condition=StoppingCondition(max_runtime_in_seconds=args.max_run),
            networking=Networking(subnets=subnets, security_group_ids=security_groups),
            output_data_config=OutputDataConfig(s3_output_path=args.output_s3) if args.output_s3 else None,
            hyperparameters=build_hyperparameters(config, args, prefixes, prod_prefixes),
            environment=environment,
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

        print("Submitting SageMaker anonymised training job")
        print(f"  image: {args.image_uri}")
        print(f"  role:  {role}")
        print(f"  instance: {args.instance_type} x {args.instance_count}")
        print(f"  volume: {args.volume_size} GB")
        print(f"  efs: {args.efs_id}:{efs_directory} -> /opt/ml/input/data/datalake")
        print(f"  prod efs: {args.efs_id}:{prod_directory} -> /opt/ml/input/data/prod")
        print(f"  vpc subnets: {subnets}")
        print(f"  security groups: {security_groups}")
        print(f"  config: {args.config}")
        print(f"  csv: {config.get('csv')}")
        if config.get("validation_csv"):
            print(f"  validation_csv: {config.get('validation_csv')}")
        if config.get("test_csv"):
            print(f"  test_csv: {config.get('test_csv')}")
        print(f"  patches: size={args.patch_size} per_image={args.patches_per_image} "
              f"zones={weights} margin={args.patch_margin}%")
        print(f"  anonymize_eval: {args.anonymize_eval}")
        if mlflow_arn:
            print(f"  mlflow: {mlflow_arn}")
            print(f"  mlflow experiment: {args.mlflow_experiment}")
        trainer.train(wait=args.wait, logs=args.wait)
        job = trainer._latest_training_job
        if job is not None:
            name = getattr(job, "training_job_name", None) or getattr(job, "name", None)
            print(f"Training job: {name}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
