#!/usr/bin/env python3
"""Launch train_efficientnet_b2.py as an Amazon SageMaker training job.

The dataset is never copied to S3. The EFS that Studio mounts at
/home/sagemaker-user/seon-data-efs is attached to the job as a FileSystemInput,
so it appears inside the container at /opt/ml/input/data/<channel>. The absolute
paths stored in the CSV `path` column (/mnt/dataefs/...) are rewritten to that
mount with the training script's --path-remap flag.

Run this from a SageMaker Studio terminal: that is where the EFS mount, the
execution role and the domain metadata live. EFS input requires VPC config, and
both the subnets and the security groups are auto-detected from the EFS mount
targets unless you pass them explicitly.

Examples:
    python training_job.py --dry-run \
        --csv /home/sagemaker-user/seon-data-efs/data/csv/train_df.csv

    python training_job.py \
        --csv /home/sagemaker-user/seon-data-efs/data/csv/train_df.csv \
        --validation-csv /home/sagemaker-user/seon-data-efs/data/csv/validation_df.csv \
        --instance-type ml.g5.2xlarge --epochs 35 --batch-size 32 \
        --cosine-decay --min-learning-rate 1e-7 --use-bbox-crop --margin 5 \
        --mixed-precision --require-gpu --comment "exp6 efs job"
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3

RESOURCE_METADATA = Path("/opt/ml/metadata/resource-metadata.json")
SOURCE_FILES = ("train_efficientnet_b2.py", "eer.py", "requirements.txt")
ENTRY_POINT = "train_efficientnet_b2.py"

METRIC_DEFINITIONS = [
    {"Name": "val_eer", "Regex": r"val_eer: ([0-9\.]+)"},
    {"Name": "val_acer", "Regex": r"val_acer: ([0-9\.]+)"},
    {"Name": "val_apcer", "Regex": r"val_apcer: ([0-9\.]+)"},
    {"Name": "val_bpcer", "Regex": r"val_bpcer: ([0-9\.]+)"},
    {"Name": "val_auc", "Regex": r"val_auc: ([0-9\.]+)"},
    {"Name": "val_loss", "Regex": r"val_loss: ([0-9\.]+)"},
    {"Name": "train_loss", "Regex": r" loss: ([0-9\.]+)"},
]


def read_resource_metadata() -> dict:
    if not RESOURCE_METADATA.exists():
        return {}
    with RESOURCE_METADATA.open("r", encoding="utf-8") as file:
        return json.load(file)


def detect_efs_id(studio_root: Path) -> str:
    mounts = Path("/proc/mounts")
    if mounts.exists():
        target = str(studio_root).rstrip("/")
        for line in mounts.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            if fields[1].rstrip("/") != target:
                continue
            match = re.search(r"fs-[0-9a-f]{8,}", fields[0])
            if match:
                return match.group(0)

    wanted = studio_root.name
    for file_system in boto3.client("efs").describe_file_systems()["FileSystems"]:
        if file_system.get("Name") == wanted:
            return file_system["FileSystemId"]

    raise SystemExit(
        f"Could not detect the EFS id behind {studio_root}. "
        "Pass it explicitly with --efs-id fs-xxxxxxxx (check `mount | grep efs` in Studio)."
    )


def efs_network_config(efs_id: str) -> tuple[list[str], list[str]]:
    efs = boto3.client("efs")
    mount_targets = efs.describe_mount_targets(FileSystemId=efs_id)["MountTargets"]
    if not mount_targets:
        raise SystemExit(f"EFS {efs_id} has no mount targets, so a training job cannot reach it.")

    subnets = [mount_target["SubnetId"] for mount_target in mount_targets]
    security_groups: list[str] = []
    for mount_target in mount_targets:
        groups = efs.describe_mount_target_security_groups(
            MountTargetId=mount_target["MountTargetId"]
        )["SecurityGroups"]
        for group in groups:
            if group not in security_groups:
                security_groups.append(group)

    return subnets, security_groups


def normalize(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.normpath(expanded))


def container_path(local_path: Path, studio_root: Path, efs_directory: str, mount_root: str) -> str:
    mounted_root = studio_root / efs_directory.strip("/")
    resolved = normalize(local_path)
    try:
        relative = resolved.relative_to(mounted_root)
    except ValueError:
        raise SystemExit(
            f"{local_path} is not inside the mounted EFS subtree {mounted_root}. "
            "Move the file under that directory, or adjust --studio-root / --efs-directory."
        ) from None
    return f"{mount_root}/{relative}" if str(relative) != "." else mount_root


def stage_source_dir(repo_root: Path) -> Path:
    staged = Path(tempfile.mkdtemp(prefix="liveness-source-"))
    for name in SOURCE_FILES:
        source = repo_root / name
        if not source.exists():
            raise SystemExit(f"Missing {source}, cannot build the job source directory.")
        shutil.copy2(source, staged / name)
    return staged


def build_hyperparameters(args: argparse.Namespace, remaps: list[str], output_dir: str) -> dict:
    hyperparameters = {
        "csv": args.container_csv,
        "output-dir": output_dir,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "learning-rate": args.learning_rate,
        "cosine-decay": str(args.cosine_decay).lower(),
        "min-learning-rate": args.min_learning_rate,
        "validation-split": args.validation_split,
        "seed": args.seed,
        "train-backbone": str(args.train_backbone).lower(),
        "require-gpu": str(args.require_gpu).lower(),
        "mixed-precision": str(args.mixed_precision).lower(),
        "eer-threshold": args.eer_threshold,
        "use-bbox-crop": str(args.use_bbox_crop).lower(),
        "margin": args.margin,
        "bbox-aug-prob": args.bbox_aug_prob,
    }

    if args.container_validation_csv:
        hyperparameters["validation-csv"] = args.container_validation_csv
    if args.container_test_csv:
        hyperparameters["test-csv"] = args.container_test_csv
    if args.comment:
        hyperparameters["comment"] = args.comment
    if remaps:
        hyperparameters["path-remap"] = ",".join(remaps)

    return hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train_efficientnet_b2.py as a SageMaker training job over EFS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--csv", type=Path, required=True, help="Training CSV, path on the Studio filesystem.")
    data.add_argument("--validation-csv", type=Path, help="Optional fixed validation CSV.")
    data.add_argument("--test-csv", type=Path, help="Optional test CSV used for the final EER.")
    data.add_argument(
        "--studio-root",
        type=Path,
        default=Path("/home/sagemaker-user/seon-data-efs"),
        help="Where the EFS is mounted in Studio.",
    )
    data.add_argument("--efs-directory", default="/", help="EFS subtree to mount into the job.")
    data.add_argument("--channel-name", default="efs", help="Input channel name, sets the container mount point.")
    data.add_argument(
        "--csv-path-prefix",
        action="append",
        default=["/mnt/dataefs"],
        help="Extra absolute prefixes used in the CSV path column that map to the EFS root. Repeatable.",
    )

    infra = parser.add_argument_group("infrastructure")
    infra.add_argument("--efs-id", help="EFS file system id. Auto-detected from the Studio mount when omitted.")
    infra.add_argument("--subnet", action="append", default=[], help="VPC subnet. Auto-detected from EFS mount targets.")
    infra.add_argument("--security-group", action="append", default=[], help="Security group. Auto-detected from EFS mount targets.")
    infra.add_argument("--role", help="SageMaker execution role ARN. Defaults to the Studio role.")
    infra.add_argument("--instance-type", default="ml.g5.2xlarge")
    infra.add_argument("--instance-count", type=int, default=1)
    infra.add_argument("--volume-size", type=int, default=100, help="EBS size in GB, only used for outputs here.")
    infra.add_argument("--image-uri", help="Custom ECR training image. Needs the sagemaker-training package installed.")
    infra.add_argument("--framework-version", default="2.18", help="Prebuilt TensorFlow container version.")
    infra.add_argument("--py-version", default="py310")
    infra.add_argument("--keras-home", help="Set KERAS_HOME, e.g. an EFS copy of ~/.keras when the VPC has no internet.")

    job = parser.add_argument_group("job")
    job.add_argument("--job-name", help="Training job name. Defaults to liveness-b2-<timestamp>.")
    job.add_argument("--s3-output", help="S3 URI for model.tar.gz and output.tar.gz. Defaults to the session bucket.")
    job.add_argument(
        "--checkpoint-s3-uri",
        help="Stream the run directory (checkpoints, tensorboard, history.csv) to this S3 prefix while training. "
        "Recommended with --spot, since /opt/ml/output/data is only uploaded when the job ends.",
    )
    job.add_argument("--max-run", type=int, default=24 * 3600, help="Job timeout in seconds.")
    job.add_argument("--spot", action="store_true", help="Use managed spot instances.")
    job.add_argument("--max-wait", type=int, default=48 * 3600, help="Spot queue + run budget in seconds.")
    job.add_argument("--keep-alive", type=int, default=0, help="Warm pool seconds, keeps the instance for the next job.")
    job.add_argument("--no-wait", action="store_true", help="Return as soon as the job is submitted.")
    job.add_argument("--dry-run", action="store_true", help="Print the resolved configuration and exit.")

    training = parser.add_argument_group("training script")
    training.add_argument("--epochs", type=int, default=12)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--learning-rate", type=float, default=1e-5)
    training.add_argument("--cosine-decay", action="store_true")
    training.add_argument("--min-learning-rate", type=float, default=0.0)
    training.add_argument("--validation-split", type=float, default=0.1)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--train-backbone", action="store_true")
    training.add_argument("--require-gpu", action="store_true")
    training.add_argument("--mixed-precision", action="store_true")
    training.add_argument("--eer-threshold", type=float, default=0.5)
    training.add_argument("--use-bbox-crop", action="store_true")
    training.add_argument("--margin", type=float, default=0.0)
    training.add_argument("--bbox-aug-prob", type=float, default=0.0)
    training.add_argument("--comment", default="", help="Free-text note saved to report.json.")

    return parser.parse_args()


def main() -> None:
    import sagemaker
    from sagemaker.inputs import FileSystemInput

    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    studio_root = normalize(args.studio_root)
    mount_root = f"/opt/ml/input/data/{args.channel_name}"
    efs_id = args.efs_id or detect_efs_id(studio_root)
    subnets, security_groups = (args.subnet, args.security_group)
    if not subnets or not security_groups:
        detected_subnets, detected_groups = efs_network_config(efs_id)
        subnets = subnets or detected_subnets
        security_groups = security_groups or detected_groups

    args.container_csv = container_path(args.csv, studio_root, args.efs_directory, mount_root)
    args.container_validation_csv = (
        container_path(args.validation_csv, studio_root, args.efs_directory, mount_root)
        if args.validation_csv
        else None
    )
    args.container_test_csv = (
        container_path(args.test_csv, studio_root, args.efs_directory, mount_root)
        if args.test_csv
        else None
    )

    remaps = [f"{studio_root}={mount_root}"]
    for prefix in args.csv_path_prefix:
        prefix = prefix.rstrip("/")
        if prefix and prefix != str(studio_root):
            remaps.append(f"{prefix}={mount_root}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"liveness-b2-{timestamp}"
    output_dir = "/opt/ml/checkpoints" if args.checkpoint_s3_uri else "/opt/ml/output/data"
    hyperparameters = build_hyperparameters(args, remaps, output_dir)

    if args.spot and not args.checkpoint_s3_uri:
        print("Warning: spot without --checkpoint-s3-uri loses every epoch checkpoint if the instance is reclaimed.")

    environment = {"TF_FORCE_GPU_ALLOW_GROWTH": "true", "PYTHONUNBUFFERED": "1"}
    if args.keras_home:
        environment["KERAS_HOME"] = args.keras_home

    resource_metadata = read_resource_metadata()
    configuration = {
        "job_name": job_name,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "efs_id": efs_id,
        "efs_directory": args.efs_directory,
        "container_mount": mount_root,
        "subnets": subnets,
        "security_groups": security_groups,
        "image_uri": args.image_uri,
        "framework_version": None if args.image_uri else args.framework_version,
        "spot": args.spot,
        "environment": environment,
        "hyperparameters": hyperparameters,
        "studio_domain": resource_metadata.get("DomainId"),
    }
    print(json.dumps(configuration, indent=2, default=str))

    if args.dry_run:
        print("\nDry run, nothing submitted.")
        return

    session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()
    source_dir = stage_source_dir(repo_root)

    common = {
        "entry_point": ENTRY_POINT,
        "source_dir": str(source_dir),
        "role": role,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "volume_size": args.volume_size,
        "max_run": args.max_run,
        "output_path": args.s3_output,
        "base_job_name": "liveness-b2",
        "hyperparameters": hyperparameters,
        "environment": environment,
        "metric_definitions": METRIC_DEFINITIONS,
        "subnets": subnets,
        "security_group_ids": security_groups,
        "sagemaker_session": session,
        "disable_profiler": True,
    }
    if args.checkpoint_s3_uri:
        common.update(checkpoint_s3_uri=args.checkpoint_s3_uri, checkpoint_local_path=output_dir)
    if args.spot:
        common.update(use_spot_instances=True, max_wait=args.max_wait)
    if args.keep_alive:
        common["keep_alive_period_in_seconds"] = args.keep_alive

    if args.image_uri:
        from sagemaker.estimator import Estimator

        estimator = Estimator(image_uri=args.image_uri, **common)
    else:
        from sagemaker.tensorflow import TensorFlow

        estimator = TensorFlow(
            framework_version=args.framework_version,
            py_version=args.py_version,
            **common,
        )

    file_system_input = FileSystemInput(
        file_system_id=efs_id,
        file_system_type="EFS",
        directory_path=args.efs_directory,
        file_system_access_mode="ro",
    )

    estimator.fit(
        {args.channel_name: file_system_input},
        job_name=job_name,
        wait=not args.no_wait,
        logs="All",
    )

    print(f"\nJob: {job_name}")
    if args.no_wait:
        print(f"Logs: aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --follow")
    else:
        print(f"Model artifacts: {estimator.model_data}")
        print(f"Reports and checkpoints: {estimator.latest_training_job.describe()['OutputDataConfig']['S3OutputPath']}")


if __name__ == "__main__":
    main()
