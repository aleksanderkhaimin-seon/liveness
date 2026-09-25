# EfficientNetB2 Training

The training script expects a CSV with two columns:

```csv
path,label
/absolute/path/to/image_001.jpg,1
relative/path/to/image_002.jpg,0
```

Relative image paths are resolved relative to the CSV file location.

## SageMaker Training Job

`train_efficientnet_b2.py` is unchanged. SageMaker Jobs use `sagemaker_job/train.py` as the container entrypoint and `sagemaker_job/launch.py` to submit the job.

Images are read from the Studio EFS data directory, which holds both `datalake/` and `processed_v3/`:

```text
/home/sagemaker-user/seon-data-efs/data
```

That directory is mounted read-only into the job at `/opt/ml/input/data/datalake`, and manifest paths are rewritten to it. All of these prefixes are recognized and map to the same files:

```text
/home/sagemaker-user/seon-data-efs/data
/mnt/custom-file-systems/efs/fs-0773949cd1f9915ec/seon-data-efs/data
/home/sagemaker-user/custom-file-systems/efs/fs-0773949cd1f9915ec/seon-data-efs/data
/mnt/dataefs/data
```

The launcher samples each manifest before submitting and fails immediately if paths fall outside the mounted directory. The job runs in the Studio VPC so it can mount the filesystem. SageMaker still uploads source code and `model.tar.gz` to the default SageMaker S3 bucket; image files are never copied.

The job uses this image:

```text
335010339905.dkr.ecr.eu-central-1.amazonaws.com/idv-ml/liveness-cuda@sha256:0bc45d1ed84f7492c3b563d562a154c5044a2eabad1d3c7642dfc6fb27446da7
```

```bash
python sagemaker_job/launch.py \
  --config configs/train.json \
  --instance-type ml.g5.xlarge
```

Train, validation, and test CSVs are taken from the config file. That file is saved into the job output as it was submitted.

VPC subnets and security groups are detected from this SageMaker domain. Override with `--subnets` and `--security-group-ids` if needed. The outbound NFS security group must be included so the training job can mount EFS.

Checkpoints, `best.keras`, `history.csv`, and `report.json` are written to `/opt/ml/model` and uploaded as `model.tar.gz`.

## SageMaker ONNX Evaluation Job

SageMaker has no separate ONNX evaluation job type. Submit a Job with `sagemaker_job/launch_eval.py`; it uses the same image and EFS mounts as training, runs `predict_onnx_csv.py`, then writes EER metrics.

From this Code Editor terminal (not the Jobs UI create form):

```bash
cd /home/sagemaker-user/seon-data-efs/users/aleksandr_khaimin/Work/Liveness/Liveness

python sagemaker_job/launch_eval.py \
  --csv data/ProdTest-0.2-val.csv \
  --onnx path/to/model.onnx \
  --instance-type ml.g4dn.xlarge
```

Several models at once:

```bash
python sagemaker_job/launch_eval.py \
  --csv data/ProdTest-0.2-val.csv \
  --onnx models/m4.onnx models/m5.onnx \
  --no-wait
```

From a Keras checkpoint (converted in the job, then scored):

```bash
python sagemaker_job/launch_eval.py \
  --csv data/ProdTest-0.2-val.csv \
  --checkpoint runs/efficientnet_b2/best.keras
```

The job appears under **Jobs → Training** with a name like `liveness-onnx-eval-...`. When it succeeds, download the output `tar.gz` (or look under the job Output in S3) for:

- `eval_report.json` (`bpcer`, `apcer`, `acer`, `eer`)
- `<model>_predictions.csv` (`path,label,score`)

ProdTest paths stay on the prod EFS mount; ONNX files are uploaded with the job source.

## Run In The Container

Start the container:

```bash
docker compose up --build
```

In another terminal:

```bash
docker compose exec liveness-training bash
python train_efficientnet_b2.py --csv test_df.csv --epochs 10 --batch-size 16
```

Or from VS Code Dev Containers, open a terminal inside the container and run:

```bash
python train_efficientnet_b2.py --csv test_df.csv --epochs 10 --batch-size 16
```

## Run With Docker Logs

If training is started with `docker compose exec`, its output is attached to that exec session and may not appear in `docker logs`. To make Docker capture training logs, run the dedicated training service:

```bash
TRAIN_CSV=train_df.csv \
VALIDATION_CSV=validation_df.csv \
TEST_CSV=extra_test_df.csv \
OUTPUT_DIR=runs/efficientnet_b2_35ep \
EPOCHS=35 \
BATCH_SIZE=32 \
COSINE_DECAY=1 \
MIN_LEARNING_RATE=1e-7 \
USE_BBOX_CROP=1 \
MARGIN=5 \
MIXED_PRECISION=1 \
REQUIRE_GPU=1 \
docker compose up liveness-train
```

Follow logs from another terminal:

```bash
docker compose logs -f liveness-train
```

The same output is also saved to:

```text
runs/logs/training.log
```

## Useful Options

```bash
python train_efficientnet_b2.py \
  --csv test_df.csv \
  --validation-csv validation_df.csv \
  --test-csv extra_test_df.csv \
  --output-dir runs/efficientnet_b2 \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --cosine-decay \
  --min-learning-rate 0.000001 \
  --use-bbox-crop \
  --margin 5 \
  --validation-split 0.2
```

`--csv` is used for training data. If `--validation-csv` is omitted, validation is split from `--csv`. `--test-csv` is optional and is used only for final testing, prediction export, and EER metrics.

At the end of every epoch, validation EER metrics are computed and logged:

- `val_bpcer`
- `val_apcer`
- `val_acer`
- `val_eer`
- `val_eer_threshold`

These appear in `history.csv` and TensorBoard.

With `--cosine-decay`, the optimizer uses cosine decay from `--learning-rate` down to `--min-learning-rate` across the requested number of epochs. The effective `learning_rate` is logged at the end of each epoch to `history.csv` and TensorBoard.

With `--use-bbox-crop`, CSV files must include a `bbox` column formatted like `[x1,y1,x2,y2]`. Cropping is applied before resize for train, validation, and test datasets. `--margin 5` expands the crop by 5% of bbox width/height on every side; `--margin -5` crops 5% inside the bbox.

Fine-tune the EfficientNetB2 backbone immediately:

```bash
python train_efficientnet_b2.py --csv test_df.csv --train-backbone
```

## Augmentations

Training uses `albumentations` in the `tf.data` input pipeline:

- horizontal flip with probability `0.5`
- affine rotation from `-10.8` to `10.8` degrees
- affine scale from `0.92` to `1.08`
- brightness multiplier from `0.8` to `1.2`

Validation, testing, checkpoint conversion, and ONNX inference do not include augmentations.

## Outputs

The script writes:

- `runs/efficientnet_b2/best.keras`
- `runs/efficientnet_b2/last.keras`
- `runs/efficientnet_b2/checkpoints/epoch_001_val_auc_....keras`
- `runs/efficientnet_b2/history.csv`
- `runs/efficientnet_b2/report.json`
- `runs/efficientnet_b2/test_predictions.csv`
- `runs/efficientnet_b2/tensorboard/`

`test_predictions.csv` contains both the raw linear `logit` and sigmoid `score`.

`report.json` contains `test_metrics` plus `test_eer_metrics`:

- `bpcer`
- `apcer`
- `acer`
- `eer`
- `eer_threshold`

## Convert Checkpoint To ONNX

Convert the best checkpoint into a model folder like `models/m1`:

```bash
python convert_checkpoint_to_onnx.py \
  runs/efficientnet_b2/best.keras \
  --output-dir models/m4
```

Overwrite an existing output folder:

```bash
python convert_checkpoint_to_onnx.py \
  runs/efficientnet_b2/best.keras \
  --output-dir models/m4 \
  --force
```

This writes:

- `models/m4/m4.onnx`
- `models/m4/m4.json`

The converter exports an inference-only ONNX graph. Training augmentation layers are not included.

## TensorBoard

Start TensorBoard from inside the container:

```bash
tensorboard --logdir runs/efficientnet_b2/tensorboard --host 0.0.0.0 --port 6006
```

Then open:

```text
http://localhost:6006
```

## Evaluate S3 CSV

For a CSV with `path,label` where `path` may be `s3://bucket/key`, sync images into the container cache and evaluate a saved checkpoint:

```bash
python eval_s3_dataframe.py data/s3_test.csv \
  --checkpoint runs/efficientnet_b2/best.keras \
  --output-dir runs/s3_eval \
  --batch-size 32 \
  --require-gpu
```

The script copies missing `s3://...` files into:

```text
/mnt/userefs/aleksandr_khaimin/Work/liveness/s3_cache
```

If many rows share a prefix, sync that prefix first:

```bash
python eval_s3_dataframe.py data/s3_test.csv \
  --checkpoint runs/efficientnet_b2/best.keras \
  --sync-prefix s3://bucket/dataset/prefix \
  --output-dir runs/s3_eval
```

Preview AWS commands without downloading:

```bash
python eval_s3_dataframe.py data/s3_test.csv \
  --checkpoint runs/efficientnet_b2/best.keras \
  --dry-run
```

With bbox crop:

```bash
python eval_s3_dataframe.py data/s3_test.csv \
  --checkpoint runs/efficientnet_b2/best.keras \
  --use-bbox-crop \
  --margin 5
```

Outputs:

- `runs/s3_eval/predictions.csv`
- `runs/s3_eval/report.json`

## Predict CSV With Checkpoint

For a local CSV in the same `path,label` format used for training:

```bash
python predict_checkpoint_csv.py data/test.csv \
  --checkpoint runs/efficientnet_b2/best.keras \
  --output-csv runs/predictions.csv \
  --batch-size 32
```

Missing files are skipped by default. Use `--on-missing raise` to fail instead.

The output CSV contains exactly:

```csv
path,label,score
```

With bbox crop:

```bash
python predict_checkpoint_csv.py data/test.csv \
  --checkpoint runs/efficientnet_b2/best.keras \
  --output-csv runs/predictions.csv \
  --use-bbox-crop \
  --margin 5
```
