#!/usr/bin/env python3
"""Download SageMaker training artifacts and render an EER-focused report.

Example:

    python visualize_sagemaker_jobs.py \\
      --s3 s3://sagemaker-eu-central-1-335010339905/liveness-efficientnet-b2/ \\
      --out eval_results
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eer import compute_eer, compute_frr_far, get_fr_fa_at_threshold

KEEP_FROM_TAR = {
    "report.json",
    "history.csv",
    "train_config.json",
    "test_predictions.csv",
    "eval_report.json",
}

LARGE_EXCLUDE_HINTS = ("manifests/train.csv", "input/code/data/train_")


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def list_s3_job_dirs(s3_uri: str) -> list[str]:
    prefix = s3_uri.rstrip("/") + "/"
    result = subprocess.run(
        ["aws", "s3", "ls", prefix],
        check=True,
        capture_output=True,
        text=True,
    )
    jobs: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "PRE":
            jobs.append(parts[1].rstrip("/"))
    return jobs


def local_job_dirs(dest: Path) -> set[str]:
    if not dest.exists():
        return set()
    return {path.name for path in dest.iterdir() if path.is_dir()}


SYNC_INCLUDES = (
    "output/model.tar.gz",
    "output/output.tar.gz",
    "output/*.json",
    "output/*.csv",
    "output/*.onnx",
    "output/converted/**",
    "input/code/configs/*",
)


def sync_job(s3_uri: str, dest: Path, job: str) -> None:
    prefix = s3_uri.rstrip("/") + "/" + job + "/"
    target = dest / job
    target.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", prefix, str(target) + "/", "--exclude", "*"]
    for pattern in SYNC_INCLUDES:
        cmd.extend(["--include", pattern])
    run(cmd)


def sync_s3(s3_uri: str, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    remote = list_s3_job_dirs(s3_uri)
    existing = local_job_dirs(dest)
    missing = [job for job in remote if job not in existing]
    print(f"s3 jobs: {len(remote)}  local: {len(existing)}  to download: {len(missing)}")
    for job in existing & set(remote):
        print(f"skip {job} (already in {dest})")
    for job in missing:
        print(f"download {job}")
        sync_job(s3_uri, dest, job)
    return missing


def _keep_tar_member(relative: Path) -> bool:
    if relative.is_absolute() or ".." in relative.parts:
        return False
    if "tensorboard" in relative.parts:
        return False
    name = relative.name
    if relative.suffix in {".keras", ".onnx"}:
        return True
    if name in KEEP_FROM_TAR or relative.suffix == ".json":
        return True
    return False


def extract_tarball(tar_path: Path, dest: Path) -> list[str]:
    extracted: list[str] = []
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = Path(member.name)
            if not _keep_tar_member(relative):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(handle.read())
            extracted.append(str(relative))
    return extracted


def extract_all(root: Path, jobs: list[str] | None = None) -> None:
    tar_paths = sorted(root.glob("*/output/model.tar.gz")) + sorted(
        root.glob("*/output/output.tar.gz")
    )
    allowed = set(jobs) if jobs is not None else None
    for tar_path in tar_paths:
        job_name = tar_path.parent.parent.name
        if allowed is not None and job_name not in allowed:
            continue
        dest = tar_path.parent
        names = extract_tarball(tar_path, dest)
        tar_path.unlink()
        print(f"extracted {job_name}: {names}")
        print(f"deleted {tar_path}")


def load_json(path: Path) -> dict | list | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path) -> list[dict[str, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def det_from_predictions(path: Path, max_points: int = 64) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            return None
        score_key = "score" if "score" in reader.fieldnames else None
        if score_key is None:
            return None
        for row in reader:
            labels.append(int(row["label"]))
            scores.append(float(row[score_key]))
    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float32)
    tar = scores_array[labels_array == 1]
    imp = scores_array[labels_array == 0]
    tar = tar[~np.isnan(tar)]
    imp = imp[~np.isnan(imp)]
    if len(tar) == 0 or len(imp) == 0:
        return None
    thresholds, frr, far = compute_frr_far(tar, imp)
    eer, eer_threshold = compute_eer(tar, imp)
    bpcer, apcer = get_fr_fa_at_threshold(tar, imp, threshold=0.5)
    step = max(1, len(frr) // max_points)
    return {
        "n_live": int(len(tar)),
        "n_attack": int(len(imp)),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "bpcer_at_0_5": float(bpcer),
        "apcer_at_0_5": float(apcer),
        "frr": (frr[::step] * 100.0).tolist(),
        "far": (far[::step] * 100.0).tolist(),
        "thresholds": thresholds[::step].tolist(),
    }


def describe_jobs(job_names: list[str], region: str) -> dict[str, dict]:
    try:
        import boto3
    except ImportError:
        return {}
    client = boto3.client("sagemaker", region_name=region)
    out: dict[str, dict] = {}
    for name in job_names:
        try:
            job = client.describe_training_job(TrainingJobName=name)
        except Exception as error:
            out[name] = {"status": "unknown", "error": str(error)}
            continue
        hps = job.get("HyperParameters") or {}
        out[name] = {
            "status": job.get("TrainingJobStatus"),
            "secondary_status": job.get("SecondaryStatus"),
            "failure_reason": (job.get("FailureReason") or "")[:400],
            "comment": hps.get("comment"),
            "train_backbone": hps.get("train_backbone"),
            "csv": hps.get("csv"),
            "validation_csv": hps.get("validation_csv"),
            "test_csv": hps.get("test_csv"),
            "creation_time": job["CreationTime"].isoformat() if job.get("CreationTime") else None,
        }
    return out


def collect_jobs(root: Path, region: str) -> list[dict]:
    jobs: list[dict] = []
    job_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("liveness-")
    )
    meta = describe_jobs([path.name for path in job_dirs], region)
    for path in job_dirs:
        output = path / "output"
        report = load_json(output / "report.json")
        if not isinstance(report, dict):
            report = {}
        eval_report = load_json(output / "eval_report.json")
        config = load_json(output / "train_config.json") or load_json(output / "train-gpu.json")
        if not isinstance(config, dict):
            config = (report.get("config") if isinstance(report.get("config"), dict) else {}) or {}
        history = load_history(output / "history.csv")
        if not history and isinstance(report.get("history"), dict):
            hist = report["history"]
            n = len(hist.get("val_eer") or [])
            history = [
                {key: values[i] for key, values in hist.items() if isinstance(values, list) and i < len(values)}
                | {"epoch": float(i)}
                for i in range(n)
            ]
        det = det_from_predictions(output / "test_predictions.csv")
        test_eer = (report.get("test_eer_metrics") or {}).get("eer")
        if test_eer is None and isinstance(eval_report, list) and eval_report:
            eers = [
                item.get("metrics", {}).get("eer")
                for item in eval_report
                if isinstance(item, dict)
            ]
            eers = [value for value in eers if value is not None]
            if eers:
                test_eer = min(eers)
        val_eer = None
        if history:
            val_eer = history[-1].get("val_eer")
        jobs.append(
            {
                "name": path.name,
                "short": path.name.split("-")[-1],
                "path": str(path),
                "sagemaker": meta.get(path.name, {}),
                "comment": report.get("comment") or config.get("comment") or meta.get(path.name, {}).get("comment"),
                "train_backbone": config.get("train_backbone"),
                "train_csv": report.get("csv") or config.get("csv"),
                "validation_csv": report.get("validation_csv") or config.get("validation_csv"),
                "test_csv": report.get("test_csv") or config.get("test_csv"),
                "train_samples": report.get("train_samples") or report.get("samples"),
                "validation_samples": report.get("validation_samples"),
                "test_samples": report.get("test_samples"),
                "completed_epochs": report.get("completed_epochs"),
                "test_metrics": report.get("test_metrics") or {},
                "test_eer_metrics": report.get("test_eer_metrics") or {},
                "eval_report": eval_report,
                "history": history,
                "det": det,
                "test_eer": test_eer,
                "val_eer": val_eer,
                "has_report": bool(report.get("test_eer_metrics") or report.get("history")),
            }
        )
    return jobs


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_plots(jobs: list[dict]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [job for job in jobs if job.get("has_report") and job.get("history")]
    images: dict[str, str] = {}
    if not completed:
        return images

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for job in completed:
        epochs = [int(row.get("epoch", i)) + 1 for i, row in enumerate(job["history"])]
        values = [row["val_eer"] for row in job["history"] if "val_eer" in row]
        ax.plot(epochs[: len(values)], values, marker="o", label=job["short"])
    ax.set_title("Validation EER by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EER (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Job")
    images["val_eer"] = _fig_to_b64(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for job in completed:
        epochs = [int(row.get("epoch", i)) + 1 for i, row in enumerate(job["history"])]
        for key, style in (("val_bpcer", "-"), ("val_apcer", "--")):
            values = [row[key] for row in job["history"] if key in row]
            if values:
                ax.plot(epochs[: len(values)], values, linestyle=style, marker="o", label=f"{job['short']} {key}")
    ax.set_title("Validation BPCER / APCER by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    images["val_bpcer_apcer"] = _fig_to_b64(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    labels = [job["short"] for job in completed if job.get("test_eer") is not None]
    values = [job["test_eer"] for job in completed if job.get("test_eer") is not None]
    ax.bar(labels, values, color="#3b6d9a")
    ax.set_title("Test-set EER (best checkpoint)")
    ax.set_xlabel("Job")
    ax.set_ylabel("EER (%)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    images["test_eer"] = _fig_to_b64(fig)
    plt.close(fig)

    det_jobs = [job for job in completed if job.get("det")]
    if det_jobs:
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        for job in det_jobs:
            ax.plot(job["det"]["far"], job["det"]["frr"], label=job["short"])
            ax.scatter([job["det"]["eer"]], [job["det"]["eer"]], s=28)
        ax.plot([0, 50], [0, 50], color="#888888", linestyle=":", linewidth=1, label="FAR = FRR")
        ax.set_title("DET on test predictions (score)")
        ax.set_xlabel("FAR / APCER (%)")
        ax.set_ylabel("FRR / BPCER (%)")
        ax.set_xlim(0, min(50, max(max(job["det"]["far"]) for job in det_jobs) * 1.05))
        ax.set_ylim(0, min(50, max(max(job["det"]["frr"]) for job in det_jobs) * 1.05))
        ax.grid(True, alpha=0.3)
        ax.legend()
        images["det"] = _fig_to_b64(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for job in completed:
        epochs = [int(row.get("epoch", i)) + 1 for i, row in enumerate(job["history"])]
        values = [row["val_auc"] for row in job["history"] if "val_auc" in row]
        if values:
            ax.plot(epochs[: len(values)], values, marker="o", label=job["short"])
    ax.set_title("Validation AUC by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Job")
    images["val_auc"] = _fig_to_b64(fig)
    plt.close(fig)

    return images


def html_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_html(jobs: list[dict], images: dict[str, str], dest: Path) -> None:
    completed = [job for job in jobs if job.get("has_report")]
    rows = []
    for job in jobs:
        sm = job.get("sagemaker") or {}
        eer = job.get("test_eer")
        val = job.get("val_eer")
        rows.append(
            "<tr>"
            f"<td>{html_escape(job['name'])}</td>"
            f"<td>{html_escape(sm.get('status'))}</td>"
            f"<td>{'' if val is None else f'{val:.3f}'}</td>"
            f"<td>{'' if eer is None else f'{eer:.3f}'}</td>"
            f"<td>{html_escape(job.get('train_csv'))}</td>"
            f"<td>{html_escape(job.get('validation_csv'))}</td>"
            f"<td>{html_escape(job.get('test_csv'))}</td>"
            f"<td>{html_escape((job.get('comment') or '')[:160])}</td>"
            "</tr>"
        )

    figures = []
    captions = {
        "val_eer": "Validation EER (%) vs epoch. Lower is better.",
        "val_bpcer_apcer": "Validation BPCER (missed live) and APCER (accepted attack) vs epoch.",
        "test_eer": "Test EER (%) from report.json after the last epoch / best checkpoint.",
        "det": "DET curve from test_predictions.csv scores. Marker is EER.",
        "val_auc": "Validation AUC vs epoch (supporting metric).",
    }
    for key, caption in captions.items():
        if key in images:
            figures.append(
                f"<figure><img src='data:image/png;base64,{images[key]}' alt='{html_escape(caption)}'/>"
                f"<figcaption>{html_escape(caption)}</figcaption></figure>"
            )

    details = []
    for job in completed:
        metrics = job.get("test_eer_metrics") or {}
        details.append(
            "<section class='job'>"
            f"<h3>{html_escape(job['name'])}</h3>"
            f"<p>{html_escape(job.get('comment'))}</p>"
            "<ul>"
            f"<li>Train samples: {html_escape(job.get('train_samples'))}</li>"
            f"<li>Val samples: {html_escape(job.get('validation_samples'))} · {html_escape(job.get('validation_csv'))}</li>"
            f"<li>Test samples: {html_escape(job.get('test_samples'))} · {html_escape(job.get('test_csv'))}</li>"
            f"<li>Backbone trained: {html_escape(job.get('train_backbone'))}</li>"
            f"<li>Test EER: {metrics.get('eer'):.4f}% @ threshold {metrics.get('eer_threshold'):.4f}</li>"
            f"<li>Test BPCER@0.5: {metrics.get('bpcer'):.4f}% · APCER@0.5: {metrics.get('apcer'):.4f}% · ACER: {metrics.get('acer'):.4f}%</li>"
            "</ul></section>"
        )

    dest.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Liveness SageMaker EER report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #1b1b1b; background: #fafafa; }}
    h1, h2, h3 {{ font-weight: 600; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f0f0; }}
    figure {{ margin: 24px 0; }}
    img {{ max-width: 100%; background: #fff; border: 1px solid #e5e5e5; }}
    figcaption {{ color: #555; font-size: 13px; margin-top: 6px; }}
    .job {{ margin: 20px 0; padding: 12px 16px; background: #fff; border: 1px solid #e5e5e5; }}
    .note {{ background: #fff8e8; border: 1px solid #ead9a8; padding: 12px 16px; }}
  </style>
</head>
<body>
  <h1>Liveness training EER</h1>
  <p>Source: SageMaker prefix <code>s3://sagemaker-eu-central-1-335010339905/liveness-efficientnet-b2/</code>.
  Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.</p>
  <p class="note">EER is the operating point where FAR (APCER) equals FRR (BPCER), in percent.
  Jobs that reuse the same CSV for validation and test will report matching val/test EER and do not measure production generalization.
  Later completed jobs validate on ProdTest and test on Pinterest — those two EER numbers are not comparable.</p>
  <h2>Jobs</h2>
  <table>
    <thead><tr><th>Job</th><th>Status</th><th>Last val EER %</th><th>Test EER %</th><th>Train CSV</th><th>Val CSV</th><th>Test CSV</th><th>Comment</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Plots</h2>
  {''.join(figures)}
  <h2>Completed job details</h2>
  {''.join(details)}
</body>
</html>
""",
        encoding="utf-8",
    )


def slim_for_json(jobs: list[dict]) -> list[dict]:
    slim = []
    for job in jobs:
        item = dict(job)
        det = item.get("det")
        if det:
            item["det"] = {
                "n_live": det["n_live"],
                "n_attack": det["n_attack"],
                "eer": det["eer"],
                "eer_threshold": det["eer_threshold"],
                "bpcer_at_0_5": det["bpcer_at_0_5"],
                "apcer_at_0_5": det["apcer_at_0_5"],
                "far": det["far"],
                "frr": det["frr"],
            }
        slim.append(item)
    return slim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync SageMaker job outputs and plot EER.")
    parser.add_argument(
        "--s3",
        default="s3://sagemaker-eu-central-1-335010339905/liveness-efficientnet-b2/",
        help="S3 prefix to sync. Skips large train manifests; downloads model.tar.gz and configs.",
    )
    parser.add_argument("--out", type=Path, default=Path("eval_results"))
    parser.add_argument("--region", default="eu-central-1")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_sync:
        sync_s3(args.s3, args.out)
    if not args.skip_extract:
        extract_all(args.out)
    jobs = collect_jobs(args.out, args.region)
    images = render_plots(jobs)
    html_path = args.html or (args.out / "training_eer_report.html")
    summary_path = args.summary or (args.out / "training_eer_summary.json")
    write_html(jobs, images, html_path)
    summary_path.write_text(json.dumps(slim_for_json(jobs), indent=2, default=str), encoding="utf-8")
    print(f"wrote {html_path}")
    print(f"wrote {summary_path}")
    completed = [job for job in jobs if job.get("test_eer") is not None]
    if completed:
        best = min(completed, key=lambda job: job["test_eer"])
        print(f"best test EER: {best['test_eer']:.4f}%  ({best['name']})")


if __name__ == "__main__":
    main()
