#!/usr/bin/env python3
"""Hyperparameter search for liveness training with SageMaker Automatic Model Tuning.

Reuses launch.build_trainer(), so image, EFS channels, VPC and MLflow are those of a normal
training job. Every trial runs sagemaker_job/train.py, which prints `objective_val_eer: <x>`
once at the end (val_eer of the epoch best.keras comes from); the tuner minimises it.

val_eer on one prod set moves 1.5-3 points with the seed alone. Search with the config's fixed
seed on a validation_csv that is NOT the set you report, then re-run the best few configs with
several seeds and report on the held-out test_csv.

    python sagemaker_job/tune.py --config configs/train-gpu.json --print-space
    python sagemaker_job/tune.py --config configs/train-gpu.json --epochs 1 --dry-run
    python sagemaker_job/tune.py --config configs/train-gpu.json --max-jobs 2 --epochs 1
    python sagemaker_job/tune.py --config configs/train-gpu.json --max-jobs 30 --max-parallel-jobs 2
    python sagemaker_job/tune.py --results <tuning-job-name>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import launch

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_NAME = "objective_val_eer"
OBJECTIVE_REGEX = r"objective_val_eer: ([0-9.]+)"
DEFAULT_MLFLOW_EXPERIMENT = "liveness-hpo"
FREQ_KEYS = ("freq_rescale_p", "freq_bandstop_p", "freq_ampmix_p")
SCALES = {"linear": "Linear", "log": "Logarithmic"}


def continuous(low: float, high: float, scale: str = "linear") -> dict:
    return {"type": "continuous", "min": low, "max": high, "scale": scale}


def categorical(*values) -> dict:
    return {"type": "categorical", "values": list(values)}


def default_space(preset: str) -> dict[str, dict]:
    """head: frozen backbone, only the classifier head trains. finetune: backbone trains too."""
    space = {
        "learning_rate": continuous(3e-5, 3e-3, "log"),
        "batch_size": categorical(32, 64),
        "rot90_prob": continuous(0.0, 0.5),
        **{key: continuous(0.0, 0.7) for key in FREQ_KEYS},
        "augment": categorical("base", "domain"),
    }
    if preset == "finetune":
        space["learning_rate"] = continuous(1e-6, 1e-4, "log")
        space["batch_size"] = categorical(16, 32)
    return space


def load_space(path: Path) -> dict[str, dict]:
    space = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(space, dict):
        raise ValueError(f"{path} must contain a JSON object {{hyperparameter: spec}}")
    return space


def validate_space(space: dict[str, dict], known: set[str], min_learning_rate: float | None) -> None:
    if not space:
        raise ValueError("The search space is empty")
    for name, spec in space.items():
        if name not in known and name not in FREQ_KEYS:
            raise ValueError(
                f"{name!r} is not a training hyperparameter (use underscores, e.g. learning_rate); "
                "train.py would ignore it and every trial would be wasted"
            )
        kind = spec.get("type")
        if kind in ("continuous", "integer"):
            low, high = spec.get("min"), spec.get("max")
            if isinstance(low, bool) or isinstance(high, bool) or not all(isinstance(v, (int, float)) for v in (low, high)):
                raise ValueError(f"{name}: min and max must be numbers")
            if not low < high:
                raise ValueError(f"{name}: min must be below max, got {low} .. {high}")
            if kind == "integer" and (int(low) != low or int(high) != high):
                raise ValueError(f"{name}: integer range needs whole numbers")
            scale = spec.get("scale", "linear")
            if scale not in SCALES:
                raise ValueError(f"{name}: scale must be one of {sorted(SCALES)}, got {scale!r}")
            if scale == "log" and low <= 0:
                raise ValueError(f"{name}: a log scale needs min > 0")
        elif kind == "categorical":
            values = spec.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"{name}: categorical needs a non-empty values list")
        else:
            raise ValueError(f"{name}: type must be continuous, integer or categorical, got {kind!r}")
    lr = space.get("learning_rate")
    if lr and lr["type"] != "categorical" and min_learning_rate is not None and lr["min"] <= min_learning_rate:
        raise ValueError(
            f"learning_rate min {lr['min']:g} is not above the static min_learning_rate {min_learning_rate:g}; "
            "cosine decay would not decay"
        )


def describe_range(spec: dict) -> str:
    if spec["type"] == "categorical":
        return "{" + ", ".join(str(v) for v in spec["values"]) + "}"
    scale = " (log)" if spec.get("scale") == "log" else ""
    return f"{spec['min']:g} .. {spec['max']:g}{scale}"


def sdk_ranges(space: dict[str, dict]) -> dict:
    from sagemaker.core.parameter import CategoricalParameter, ContinuousParameter, IntegerParameter

    ranges = {}
    for name, spec in space.items():
        if spec["type"] == "categorical":
            ranges[name] = CategoricalParameter(list(spec["values"]))
            continue
        cls, cast = (ContinuousParameter, float) if spec["type"] == "continuous" else (IntegerParameter, int)
        ranges[name] = cls(cast(spec["min"]), cast(spec["max"]), scaling_type=SCALES[spec.get("scale", "linear")])
    return ranges


def as_channels(trainer) -> list:
    """trainer.input_data_config with every InputData turned into a Channel.

    ModelTrainer.train() does this conversion itself. The tuner bypasses it and copies the list
    into its request as-is, and that request only accepts Channel (pydantic ValidationError).
    """
    from sagemaker.core.training.configs import InputData

    channels = []
    for item in trainer.input_data_config or []:
        if not isinstance(item, InputData):
            channels.append(item)
            continue
        channel = trainer.create_input_data_channel(item.channel_name, item.data_source)
        if item.content_type:
            channel.content_type = item.content_type
        channels.append(channel)
    return channels


def _field(obj, *names):
    """Value of the first present attribute/key; the SDK returns objects or dicts by version."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def check_vpc(tuner) -> str | None:
    """Return the tuning job name. The SDK drops the VPC config silently if building it fails, and
    without it no trial can mount EFS, so stop the job right away instead of paying for failures."""
    try:
        description = tuner.describe()
    except Exception as error:
        print(f"Could not describe the tuning job to check its VPC config: {error}")
        return getattr(tuner, "_current_job_name", None)
    name = _field(description, "hyper_parameter_tuning_job_name", "HyperParameterTuningJobName") or getattr(
        tuner, "_current_job_name", None
    )
    definition = _field(description, "training_job_definition", "TrainingJobDefinition")
    if definition is None:
        print("Could not read the training job definition; check its VPC config in the console.")
    elif _field(definition, "vpc_config", "VpcConfig") is None:
        tuner.stop_tuning_job()
        raise SystemExit(
            f"Tuning job {name} was submitted without a VPC config, so the trials cannot mount EFS. "
            "It has been stopped."
        )
    return name


def trial_rows(tuning_job_name: str) -> list[dict]:
    import boto3

    client = boto3.client("sagemaker")
    paginator = client.get_paginator("list_training_jobs_for_hyper_parameter_tuning_job")
    rows = []
    for page in paginator.paginate(HyperParameterTuningJobName=tuning_job_name):
        for job in page["TrainingJobSummaries"]:
            metric = (job.get("FinalHyperParameterTuningJobObjectiveMetric") or {}).get("Value")
            start, end = job.get("TrainingStartTime"), job.get("TrainingEndTime")
            rows.append(
                {
                    "training_job": job["TrainingJobName"],
                    "status": job["TrainingJobStatus"],
                    OBJECTIVE_NAME: metric,
                    "minutes": round((end - start).total_seconds() / 60, 1) if start and end else None,
                    **job.get("TunedHyperParameters", {}),
                }
            )
    rows.sort(key=lambda row: (row[OBJECTIVE_NAME] is None, row[OBJECTIVE_NAME] or math.inf))
    return rows


def show_results(args: argparse.Namespace) -> None:
    rows = trial_rows(args.results)
    if not rows:
        raise SystemExit(f"No training jobs found for tuning job {args.results!r}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    out = args.results_csv or REPO_ROOT / "runs" / "hpo" / f"{args.results}_trials.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    shown = [c for c in columns if c != "training_job"]
    print("\t".join(["rank"] + shown))
    for rank, row in enumerate(rows, start=1):
        print("\t".join([str(rank)] + [str(row.get(c, "")) for c in shown]))
    print(f"\n{len(rows)} trials -> {out}")
    print("Next: re-run the top few configs with several seeds before trusting the ranking.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[launch.build_parser(add_help=False)],
    )
    parser.add_argument("--preset", choices=("head", "finetune"), default="head",
                        help="Default search space. head: frozen backbone (sets --no-train-backbone). "
                        "finetune: backbone trains, so lower learning rates (sets --train-backbone).")
    parser.add_argument("--space", type=Path, default=None,
                        help='JSON file {name: {"type": "continuous"|"integer"|"categorical", "min", "max", '
                        '"scale": "linear"|"log", "values"}} replacing the preset space.')
    parser.add_argument("--max-jobs", type=int, default=30, help="Number of trials.")
    parser.add_argument("--max-parallel-jobs", type=int, default=2,
                        help="Trials running at once. Bayesian search learns less the more of them run in parallel.")
    parser.add_argument("--strategy", choices=("Bayesian", "Random"), default="Bayesian")
    parser.add_argument("--tuner-seed", type=int, default=None,
                        help="Seed of the search strategy. The training seed stays the config's (or --seed).")
    parser.add_argument("--base-tuning-job-name", default="liveness-hpo")
    parser.add_argument("--tuning-max-runtime", type=int, default=None,
                        help="Cap on the whole tuning job in seconds. --max-run caps each trial.")
    parser.add_argument("--print-space", action="store_true",
                        help="Print the search space and exit. Needs no AWS access.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the trainer and tuner, print them, but do not submit. The tuning "
                        "request itself is only built inside tune(), so errors there show on a real run.")
    parser.add_argument("--results", metavar="TUNING_JOB_NAME", default=None,
                        help="List the trials of a tuning job, best first, and save them as CSV.")
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.set_defaults(mlflow_experiment=DEFAULT_MLFLOW_EXPERIMENT, wait=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.results:
        show_results(args)
        return
    if args.max_parallel_jobs > args.max_jobs:
        parser.error("--max-parallel-jobs cannot exceed --max-jobs")

    config = launch.load_train_config(args.config)
    if not config.get("validation_csv"):
        raise SystemExit(
            f"{args.config} has no validation_csv. The objective is val_eer, so it needs its own "
            "validation set rather than a split of the training CSV."
        )
    if config.get("validation_csv") == config.get("test_csv"):
        print("WARNING: validation_csv and test_csv are the same file; the final number would not be held out.")

    if args.space is not None:
        space = load_space(args.space)
    else:
        space = default_space(args.preset)
        backbone = args.preset == "finetune"
        if args.train_backbone is None:
            args.train_backbone = backbone
        elif args.train_backbone != backbone:
            print(f"WARNING: the train-backbone flag overrides --preset {args.preset}; its learning-rate range may not fit.")
    min_lr = config.get("min_learning_rate")
    known = {action.dest for action in parser._actions}
    validate_space(space, known, float(min_lr) if min_lr not in (None, "") else None)

    print("Search space:")
    for name, spec in space.items():
        print(f"  {name:<18} {spec['type']:<12} {describe_range(spec)}")
    print(f"Objective: minimise {OBJECTIVE_NAME} (val_eer on {config['validation_csv']})")
    print(f"Held out:  test_csv {config.get('test_csv')}, never part of the objective")
    waves = math.ceil(args.max_jobs / args.max_parallel_jobs)
    print(
        f"Trials: {args.max_jobs} ({args.strategy}), {args.max_parallel_jobs} at a time on "
        f"{args.instance_type}; worst case {waves} x {args.max_run / 3600:g} h wall time"
    )
    if args.print_space:
        return

    # freq_aug and freq_*_p are mutually exclusive in train.py; the tuned knobs win.
    drop = ("freq_aug",) if any(key in space for key in FREQ_KEYS) else ()
    trainer, stage = launch.build_trainer(
        args, header="Configuring SageMaker tuning job (all trials share this trainer)", drop_hyperparameters=drop
    )
    try:
        from sagemaker.train.tuner import HyperparameterTuner

        trainer.input_data_config = as_channels(trainer)
        tuner = HyperparameterTuner(
            model_trainer=trainer,
            objective_metric_name=OBJECTIVE_NAME,
            objective_type="Minimize",
            metric_definitions=[{"Name": OBJECTIVE_NAME, "Regex": OBJECTIVE_REGEX}],
            hyperparameter_ranges=sdk_ranges(space),
            strategy=args.strategy,
            max_jobs=args.max_jobs,
            max_parallel_jobs=args.max_parallel_jobs,
            max_runtime_in_seconds=args.tuning_max_runtime,
            base_tuning_job_name=args.base_tuning_job_name,
            random_seed=args.tuner_seed,
        )
        if args.dry_run:
            print("Dry run: nothing was submitted.")
            return
        # The source tarball is uploaded inside tune(), so the stage must outlive this call.
        tuner.tune(wait=False)
        name = check_vpc(tuner)
        print(f"Tuning job: {name}")
        print(f"Trials:     python sagemaker_job/tune.py --results {name}")
        if args.wait:
            tuner.wait()
            print(f"Best training job: {tuner.best_training_job()}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
