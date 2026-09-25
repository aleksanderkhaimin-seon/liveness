#!/usr/bin/env python3
"""Submit EfficientNetB2 training as a SageMaker Training Job.

Training images stay on the Studio EFS datalake. Manifest CSVs and source
code are staged from this repo; SageMaker still writes model.tar.gz to S3.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes.shapes import FileSystemDataSource, OutputDataConfig, StoppingCondition
from sagemaker.core.training.configs import Compute, InputData, Networking, SourceCode
from sagemaker.train import ModelTrainer


DEFAULT_IMAGE = (
    "335010339905.dkr.ecr.eu-central-1.amazonaws.com/idv-ml/liveness-cuda"
    "@sha256:0bc45d1ed84f7492c3b563d562a154c5044a2eabad1d3c7642dfc6fb27446da7"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFS_ID = "fs-0773949cd1f9915ec"
# Parent of both datalake/ and processed_v3/, so any manifest under data/ resolves.
DEFAULT_DATALAKE_HOST = "/home/sagemaker-user/seon-data-efs/data"
# Parent of public and internal prod buckets so document_check and extracted-frames both resolve.
DEFAULT_PROD_HOST = "/home/sagemaker-user/prod-data-efs/buckets"
EFS_MOUNT_MARKERS = (
    "/mnt/custom-file-systems/efs",
    "/home/sagemaker-user/custom-file-systems/efs",
)
METADATA_PATH = Path("/opt/ml/metadata/resource-metadata.json")


def load_train_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def resolve_repo_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.is_file() else None


def repo_relative(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError:
        return Path(path.name)


def staged_csv_relative(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return Path("data") / path.name
    return path


GPU_INSTANCE_FAMILIES = ("p2", "p3", "p4", "p4d", "p5", "g4dn", "g5", "g5g", "g6", "g6e")
DEFAULT_MLFLOW_EXPERIMENT = "liveness-efficientnet-b2"


def instance_has_gpu(instance_type: str) -> bool:
    parts = instance_type.split(".")
    family = parts[1] if len(parts) > 1 else ""
    return family in GPU_INSTANCE_FAMILIES


def stringify(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    # SageMaker writes hyperparameters into a bash env file. Quotes and $ break `source`.
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text.replace("'", "").replace('"', "").replace("`", "").replace("$", "")


def efs_directory_path(host_path: str, file_system_id: str) -> str:
    host = str(Path(host_path).resolve())
    for marker in EFS_MOUNT_MARKERS:
        root = f"{marker}/{file_system_id}"
        if host == root or host.startswith(root + "/"):
            relative = host[len(root) :] or "/"
            return relative
    raise ValueError(
        f"Host datalake path {host} is not under an EFS mount for {file_system_id}. "
        f"Expected one of: {[f'{m}/{file_system_id}' for m in EFS_MOUNT_MARKERS]}"
    )


def strip_prefix_aliases(efs_directory: str, file_system_id: str) -> list[str]:
    """Host paths that refer to the mounted EFS directory, in every form seen in manifests."""
    relative = efs_directory.strip("/")
    aliases = [f"{marker}/{file_system_id}/{relative}" for marker in EFS_MOUNT_MARKERS]
    aliases.append(f"/home/sagemaker-user/{relative}")
    access_point, _, remainder = relative.partition("/")
    if remainder:
        # EC2 mounted the same filesystem at /mnt/dataefs.
        aliases.append(f"/mnt/dataefs/{remainder}")
    return aliases


def check_manifest_paths(csv_path: Path, prefixes: list[str], sample: int = 20) -> None:
    import csv as csv_module

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv_module.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a path column")
        unmatched = 0
        checked = 0
        for row_number, row in enumerate(reader, start=2):
            if row_number > sample + 1:
                break
            image_path = (row.get("path") or "").strip()
            checked += 1
            if not any(image_path == p or image_path.startswith(p + "/") for p in prefixes):
                unmatched += 1

    if unmatched:
        raise RuntimeError(
            f"{csv_path.name}: {unmatched}/{checked} sampled paths are not under a mounted EFS prefix. "
            "Adjust --datalake-host-path, --prod-host-path, or --strip-path-prefix."
        )


MLFLOW_APP_READY = {"Created", "Updated"}


def current_user_profile() -> str | None:
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("UserProfileName")
    except Exception:
        return None


def list_mlflow_targets() -> list[dict]:
    sagemaker = boto3.client("sagemaker")
    targets: list[dict] = []
    try:
        for item in sagemaker.list_mlflow_apps().get("Summaries") or []:
            if item.get("Status") not in MLFLOW_APP_READY:
                continue
            owner = None
            try:
                detail = sagemaker.describe_mlflow_app(Arn=item["Arn"])
                owner = (detail.get("CreatedBy") or {}).get("UserProfileName")
            except Exception:
                pass
            targets.append(
                {
                    "kind": "app",
                    "name": item.get("Name"),
                    "arn": item.get("Arn"),
                    "status": item.get("Status"),
                    "owner": owner,
                }
            )
    except Exception as error:
        print(f"Could not list MLflow apps: {error}")
    try:
        paginator = sagemaker.get_paginator("list_mlflow_tracking_servers")
        for page in paginator.paginate():
            for item in page.get("TrackingServerSummaries") or []:
                if item.get("IsActive") not in (None, "Active") and item.get("TrackingServerStatus") != "Created":
                    continue
                targets.append(
                    {
                        "kind": "tracking-server",
                        "name": item.get("TrackingServerName"),
                        "arn": item.get("TrackingServerArn"),
                        "status": item.get("TrackingServerStatus"),
                        "owner": None,
                    }
                )
    except Exception as error:
        print(f"Could not list MLflow tracking servers: {error}")
    return targets


def detect_mlflow_tracking_arn(explicit: str | None = None, app_name: str | None = None) -> str | None:
    if explicit and explicit.startswith("arn:"):
        return explicit
    name = app_name or (explicit if explicit and not explicit.startswith("arn:") else None)
    try:
        targets = list_mlflow_targets()
    except Exception as error:
        print(f"Could not list MLflow targets: {error}")
        return None
    if name:
        matches = [item for item in targets if item.get("name") == name or item.get("arn") == name]
        if not matches:
            print(f"MLflow target {name!r} not found. Available:")
            for item in targets:
                print(f"  {item['kind']} {item['name']} {item['arn']}")
            return None
        chosen = matches[0]
        print(f"MLflow: using {chosen['kind']} {chosen['name']}")
        return chosen["arn"]
    if not targets:
        return None
    user = current_user_profile()
    owned = [item for item in targets if user and item.get("owner") == user]
    pool = owned or targets
    if len(pool) == 1:
        chosen = pool[0]
        print(f"MLflow: using {chosen['kind']} {chosen['name']}")
        return chosen["arn"]
    print("Multiple MLflow apps/servers found. Pass --mlflow-app NAME or --mlflow-tracking-arn ARN.")
    for item in targets:
        owner = item.get("owner") or "-"
        print(f"  {item['kind']} {item['name']} owner={owner} {item['arn']}")
    return None


def detect_networking() -> tuple[list[str], list[str]]:
    subnets: list[str] = []
    security_groups: list[str] = []
    try:
        sagemaker = boto3.client("sagemaker")
        domain_id = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["DomainId"]
        domain = sagemaker.describe_domain(DomainId=domain_id)
        subnets = list(domain.get("SubnetIds") or [])
        security_groups = list((domain.get("DefaultUserSettings") or {}).get("SecurityGroups") or [])
        vpc_id = domain.get("VpcId")
        if vpc_id:
            ec2 = boto3.client("ec2")
            groups = ec2.describe_security_groups(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    {"Name": "group-name", "Values": [f"security-group-for-outbound-nfs-{domain_id}"]},
                ]
            ).get("SecurityGroups", [])
            for group in groups:
                if group["GroupId"] not in security_groups:
                    security_groups.append(group["GroupId"])
    except Exception as error:
        print(f"Could not auto-detect VPC settings: {error}")
    return subnets, security_groups


def config_csv_paths(config: dict) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    train_csv = resolve_repo_path(config.get("csv"))
    if train_csv is None:
        raise FileNotFoundError(f"config csv not found: {config.get('csv')}")
    resolved["csv"] = train_csv
    for key in ("validation_csv", "test_csv"):
        path = resolve_repo_path(config.get(key))
        if config.get(key) not in (None, "") and path is None:
            raise FileNotFoundError(f"config {key} not found: {config.get(key)}")
        if path is not None:
            resolved[key] = path
    return resolved


def stage_source(config_path: Path, config: dict, csv_paths: dict[str, Path]) -> tuple[Path, Path]:
    stage = Path(tempfile.mkdtemp(prefix="liveness-sagemaker-"))
    shutil.copy2(REPO_ROOT / "train_efficientnet_b2.py", stage / "train_efficientnet_b2.py")
    shutil.copy2(REPO_ROOT / "eer.py", stage / "eer.py")
    job_dir = stage / "sagemaker_job"
    job_dir.mkdir()
    shutil.copy2(REPO_ROOT / "sagemaker_job" / "train.py", job_dir / "train.py")

    config_rel = repo_relative(config_path)
    staged_config = stage / config_rel
    staged_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, staged_config)

    for key, source in csv_paths.items():
        relative = staged_csv_relative(config[key])
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return stage, config_rel


def build_hyperparameters(
    config: dict, args: argparse.Namespace, prefixes: list[str], prod_prefixes: list[str]
) -> dict[str, str]:
    hyperparameters: dict[str, str] = {}
    passthrough = (
        "epochs",
        "batch_size",
        "learning_rate",
        "cosine_decay",
        "min_learning_rate",
        "validation_split",
        "seed",
        "train_backbone",
        "require_gpu",
        "mixed_precision",
        "eer_threshold",
        "use_bbox_crop",
        "margin",
        "bbox_aug_prob",
        "degrade",
        "comment",
    )
    for key in passthrough:
        if key in config and config[key] not in (None, ""):
            hyperparameters[key] = stringify(config[key])

    cli_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "cosine_decay": args.cosine_decay,
        "min_learning_rate": args.min_learning_rate,
        "validation_split": args.validation_split,
        "seed": args.seed,
        "train_backbone": args.train_backbone,
        "require_gpu": args.require_gpu,
        "mixed_precision": args.mixed_precision,
        "eer_threshold": args.eer_threshold,
        "use_bbox_crop": args.use_bbox_crop,
        "margin": args.margin,
        "bbox_aug_prob": args.bbox_aug_prob,
        "degrade": args.degrade,
        "comment": args.comment,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            hyperparameters[key] = stringify(value)

    hyperparameters["config"] = stringify(Path(args.config_rel).as_posix())
    hyperparameters["strip_path_prefix"] = args.strip_path_prefix or "|".join(prefixes)
    if prod_prefixes:
        hyperparameters["prod_strip_path_prefix"] = "|".join(prod_prefixes)
    if args.image_root:
        hyperparameters["image_root"] = args.image_root

    has_gpu = instance_has_gpu(args.instance_type)
    if "require_gpu" not in hyperparameters:
        hyperparameters["require_gpu"] = "true" if has_gpu else "false"
    if not has_gpu:
        if stringify(hyperparameters.get("require_gpu")) == "true":
            raise RuntimeError(
                f"require_gpu is true but {args.instance_type} has no GPU. "
                "Set require_gpu to false in the config or pass --no-require-gpu."
            )
        if stringify(hyperparameters.get("mixed_precision")) == "true":
            print(f"{args.instance_type} has no GPU: disabling mixed precision.")
            hyperparameters["mixed_precision"] = "false"
    return hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch liveness training as a SageMaker Training Job.")
    parser.add_argument("--image-uri", default=DEFAULT_IMAGE)
    parser.add_argument("--role", default=None, help="IAM role ARN. Defaults to the current SageMaker execution role.")
    parser.add_argument("--instance-type", default="ml.g5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--volume-size", type=int, default=50, help="EBS volume size in GB for code and checkpoints.")
    parser.add_argument("--max-run", type=int, default=86400, help="Max runtime in seconds.")
    parser.add_argument("--base-job-name", default="liveness-efficientnet-b2")
    parser.add_argument("--output-s3", default=None, help="s3://bucket/prefix for model.tar.gz. Default session bucket.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train.json",
        help="JSON with csv, validation_csv, test_csv, and training settings. Copied into the job output as-is.",
    )
    parser.add_argument("--efs-id", default=DEFAULT_EFS_ID)
    parser.add_argument("--datalake-host-path", default=DEFAULT_DATALAKE_HOST)
    parser.add_argument("--prod-host-path", default=DEFAULT_PROD_HOST)
    parser.add_argument("--efs-directory-path", default=None, help="Path on the EFS filesystem. Inferred from --datalake-host-path when omitted.")
    parser.add_argument(
        "--strip-path-prefix",
        default=None,
        help="Pipe-separated path prefixes to rewrite onto the EFS datalake channel.",
    )
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
    parser.add_argument("--use-bbox-crop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--bbox-aug-prob", type=float)
    parser.add_argument(
        "--degrade",
        default=None,
        help="Anonymisation degradations for train_efficientnet_b2.py, e.g. mask:0.05,downscale:192. "
        "Overrides the config's degrade key.",
    )
    parser.add_argument("--comment", default=None)
    parser.add_argument("--subnets", nargs="*", default=None)
    parser.add_argument("--security-group-ids", nargs="*", default=None)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mlflow-tracking-arn",
        default=None,
        help="MLflow app or tracking-server ARN, or an app name such as ah-exp.",
    )
    parser.add_argument(
        "--mlflow-app",
        default=None,
        help="MLflow app name (for example ah-exp). Default: the app created by this Studio user, if unique.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help="MLflow experiment name inside the app.",
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log the job to SageMaker MLflow when an app or tracking server is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"--config not found: {args.config}")
    config = load_train_config(args.config)
    csv_paths = config_csv_paths(config)
    args.csv = csv_paths["csv"]
    args.validation_csv = csv_paths.get("validation_csv")
    args.test_csv = csv_paths.get("test_csv")

    session = Session()
    role = args.role or get_execution_role()
    efs_directory = args.efs_directory_path or efs_directory_path(args.datalake_host_path, args.efs_id)
    prefixes = (
        args.strip_path_prefix.split("|")
        if args.strip_path_prefix
        else strip_prefix_aliases(efs_directory, args.efs_id)
    )
    prod_directory = efs_directory_path(args.prod_host_path, args.efs_id)
    prod_prefixes = strip_prefix_aliases(prod_directory, args.efs_id)
    # Studio exposes prod as /home/sagemaker-user/prod-data-efs/... via a symlink.
    prod_host = str(Path(args.prod_host_path))
    if prod_host.rstrip("/") not in prod_prefixes:
        prod_prefixes.append(prod_host.rstrip("/"))
    combined_prefixes = prefixes + prod_prefixes
    for manifest in (args.csv, args.validation_csv, args.test_csv):
        if manifest is not None:
            check_manifest_paths(manifest, combined_prefixes)

    detected_subnets, detected_sgs = detect_networking()
    subnets = args.subnets or detected_subnets
    security_groups = args.security_group_ids or detected_sgs
    if not subnets or not security_groups:
        raise RuntimeError(
            "SageMaker must run in the EFS VPC. Pass --subnets and --security-group-ids "
            "(include the Studio default SG and the outbound-NFS SG)."
        )

    mlflow_arn = None
    if args.mlflow:
        mlflow_arn = detect_mlflow_tracking_arn(args.mlflow_tracking_arn, args.mlflow_app)
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
            environment["MLFLOW_EXPERIMENT_NAME"] = stringify(args.mlflow_experiment)
        trainer = ModelTrainer(
            training_image=args.image_uri,
            role=role,
            sagemaker_session=session,
            base_job_name=args.base_job_name,
            source_code=SourceCode(
                source_dir=str(stage),
                entry_script="sagemaker_job/train.py",
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

        print("Submitting SageMaker training job")
        print(f"  image: {args.image_uri}")
        print(f"  role:  {role}")
        print(f"  instance: {args.instance_type} x {args.instance_count}")
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
