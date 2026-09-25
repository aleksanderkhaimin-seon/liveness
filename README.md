# Liveness

EfficientNetB2 document liveness training and ONNX evaluation.

Local Docker training, checkpoints, and TensorBoard are documented in [TRAINING.md](TRAINING.md). This README covers **SageMaker Jobs**.

## SageMaker Jobs

Submit jobs from a **Code Editor terminal** in this repo. Do not use the Jobs UI **Create** form; that path skips the EFS mounts and path rewriting.

Jobs appear afterwards under **Jobs → Training**.

Image:

```text
335010339905.dkr.ecr.eu-central-1.amazonaws.com/idv-ml/liveness-cuda@sha256:0bc45d1ed84f7492c3b563d562a154c5044a2eabad1d3c7642dfc6fb27446da7
```

Working directory:

```bash
cd /home/sagemaker-user/seon-data-efs/users/aleksandr_khaimin/Work/Liveness/Liveness
```

Images stay on EFS. The job still writes source and `model.tar.gz` to the default SageMaker S3 bucket.

### Training

`train_efficientnet_b2.py` is unchanged. `sagemaker_job/launch.py` submits a job that runs `sagemaker_job/train.py`.

Settings come from the config file (`csv`, `validation_csv`, `test_csv`, epochs, LR, bbox crop). The same config is copied into the job output unchanged.

ml.g6.4xlarge - L4

```bash
python sagemaker_job/launch.py \
  --config configs/train.json \
  --instance-type ml.g6.4xlarge
```

Return immediately and watch the job in the UI:

```bash
python sagemaker_job/launch.py \
  --config configs/train.json \
  --instance-type ml.g6.4xlarge \
  --no-wait
```

Once the job exists in SageMaker, you can close the terminal. `--wait` (default) only streams logs locally.

Outputs in `model.tar.gz`:

- `best.keras`, `last.keras`, checkpoints
- `history.csv`, `report.json`, `test_predictions.csv`
- the submitted config (`train.json` / `train-gpu.json` and `train_config.json`), with the original train/val/test CSV names

CSV `path` values must sit under one of the mounted prefixes:

| Data | Host path | Job mount |
| --- | --- | --- |
| Train / processed_v3 | `/home/sagemaker-user/seon-data-efs/data` | `/opt/ml/input/data/datalake` |
| ProdTest | `/home/sagemaker-user/prod-data-efs/buckets` (public + internal) | `/opt/ml/input/data/prod` |

Studio, custom-file-systems, and `/mnt/dataefs/data` aliases of the same files are rewritten inside the job.

Avoid apostrophes in `comment`; they used to break the job env file (the launcher now strips quotes).

#### CPU-only training

Set `"require_gpu": false` in `configs/train.json` and pick a CPU instance:

```bash
python sagemaker_job/launch.py \
  --config configs/train.json \
  --instance-type ml.m5.4xlarge
```

On a non-GPU instance the launcher disables `mixed_precision` (it only helps on NVIDIA GPUs) and refuses to submit if `require_gpu` is still true. Expect CPU training to be far slower than `ml.g4dn.xlarge`.

#### MLflow

Jobs log to a SageMaker **MLflow app** (this account uses apps, not classic tracking servers). `ah-exp` is Created and owned by this Studio user, so a normal submit picks it up.

```bash
python sagemaker_job/launch.py \
  --config configs/train-gpu.json \
  --instance-type ml.g6.4xlarge
```

Or pin it:

```bash
python sagemaker_job/launch.py --config configs/train-gpu.json --mlflow-app ah-exp
python sagemaker_job/launch.py --config configs/train-gpu.json --mlflow-tracking-arn arn:aws:sagemaker:eu-central-1:335010339905:mlflow-app/app-5TWU4M5O3LJU
python sagemaker_job/launch.py --config configs/train-gpu.json --no-mlflow
```

After training, metrics from `history.csv` / `report.json` go to experiment `liveness-efficientnet-b2` in **ah-exp**. Keras checkpoints stay in `model.tar.gz`.

### ONNX evaluation

SageMaker has no separate ONNX eval job type. `sagemaker_job/launch_eval.py` submits a Training Job that runs `predict_onnx_csv.py` and writes EER metrics.

```bash
python sagemaker_job/launch_eval.py \
  --csv data/ProdTest-0.2-val.csv \
  --onnx path/to/model.onnx \
  --instance-type ml.g4dn.xlarge
```

Several models:

```bash
python sagemaker_job/launch_eval.py \
  --csv data/ProdTest-0.2-val.csv \
  --onnx models/m4.onnx models/m5.onnx \
  --no-wait
```

From a Keras checkpoint (converted in the job with `convert_checkpoint_to_onnx.py`, then scored):

```bash
python sagemaker_job/launch_eval.py \
  --csv data/ProdTest-0.2-val.csv \
  --checkpoint runs/efficientnet_b2_haia2/exp_2/best.keras \
  --instance-type ml.g6.4xlarge
```

A directory of `.keras` files is expanded non-recursively (immediate children only):

```bash
python sagemaker_job/launch_eval.py \
  --csv data/ProdTest-0.2-val.csv \
  --checkpoint runs/efficientnet_b2_haia2/exp_2
```

You can mix `--onnx` and `--checkpoint`. Converted ONNX is written to the job output as `converted/<name>/<name>.onnx`.

`--csv` defaults to `data/ProdTest-0.2-val.csv`. ONNX files are uploaded with the job source. ProdTest images stay on the prod EFS mount.

`--device auto` (default) uses onnxruntime on CPU unless the instance actually has a GPU, so CPU instances work without extra flags:

```bash
python sagemaker_job/launch_eval.py \
  --csv data/ProdTest-0.2-val.csv \
  --onnx models/exp_39.onnx \
  --instance-type ml.m5.4xlarge
```

Job output (`output/data` in S3, also packed with the model artifact):

- `eval_report.json` — `bpcer`, `apcer`, `acer`, `eer`; for Keras jobs also `checkpoint`, `checkpoint_path`, and `checkpoint_arg`
- `<model>_predictions.csv` — `path,label,score`
- `<name>.onnx` and `converted/<name>/` — converted models when `--checkpoint` was used (also inside `model.tar.gz`)

### ProdTest val CSV

`data/ProdTest-0.2.csv` uses `s3_object_key`. On the prod EFS the usable images are `merchant/session/extracted-frames/*.jpeg`. Rebuild the training-format val manifest with:

```bash
python data_prep/prepare_prodtest_val.py \
  --input-csv data/ProdTest-0.2.csv \
  --mount-root /home/sagemaker-user/prod-data-efs/buckets/id-verification-internal-prod-eu-west-1-847433666304 \
  --output-csv data/ProdTest-0.2-val.csv
```

### Where to look after submit

1. **Jobs → Training** — status, CloudWatch logs, failure reason
2. Job **Output** S3 prefix — `model.tar.gz` and extra output data
3. Hyperparameters and instance type — job details page
4. `visualize_sagemaker_jobs.py` — local EER report from those S3 prefixes (below)

VPC subnets and security groups are taken from this Studio domain (default SG plus outbound NFS). Override with `--subnets` and `--security-group-ids` if the EFS mount fails.

### Visualize training EER

`visualize_sagemaker_jobs.py` lists job folders under the SageMaker S3 prefix, downloads any that are not already in `eval_results/`, unpacks metrics, Keras checkpoints (`best.keras`, `last.keras`, `checkpoints/`), and `.onnx` files from `model.tar.gz` / `output.tar.gz`, deletes the tarball, and writes an EER-focused HTML report. TensorBoard event files are not extracted.

It skips large train manifests and does not re-download a job whose folder already exists locally. Delete that folder to force a refresh.

```bash
python visualize_sagemaker_jobs.py \
  --s3 s3://sagemaker-eu-central-1-335010339905/liveness-efficientnet-b2/ \
  --out eval_results
```

Outputs:

- `eval_results/<job-name>/output/` — extracted metrics, `checkpoints/`, and `.onnx` (the tarball is removed after unpack)
- `eval_results/training_eer_report.html` — val/test EER, BPCER/APCER, DET from test scores
- `eval_results/training_eer_summary.json` — the same numbers as JSON

Rebuild the report from files already in `eval_results/` (no S3):

```bash
python visualize_sagemaker_jobs.py --skip-sync --out eval_results
```

Cursor does not render the HTML as a page when you open the file. Serve it and use Simple Browser (`Ctrl+Shift+P` → **Simple Browser: Show**):

```bash
python -m http.server 8765 --directory eval_results
```

Then open `http://localhost:8765/training_eer_report.html`. A `file://` URL must be the absolute path (three slashes) and is often blocked in Simple Browser.
