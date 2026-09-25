# EfficientNetB2 Training

The training script expects a CSV with two columns:

```csv
path,label
/absolute/path/to/image_001.jpg,1
relative/path/to/image_002.jpg,0
```

Relative image paths are resolved relative to the CSV file location.

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

## Anonymisation Degradations

`--degrade` applies content-destroying transforms to the decoded frame before any bbox crop and before the resize to 512. They run identically on train, validation and test and on both classes, so they cannot become a label shortcut. Use them to measure how much EER survives a given anonymisation before building the export for it.

```bash
python train_efficientnet_b2.py --csv train.csv --degrade mask:0.05
python train_efficientnet_b2.py --csv train.csv --degrade mask:0.05,downscale:192
DEGRADE=mask:0.05 docker compose up liveness-train
python training_job.py --csv ... --degrade mask:0.05
```

| spec | effect | needs bbox |
|---|---|---|
| `mask:BAND` | fill the document interior with its per-channel mean, keeping a border band `BAND` × bbox size on each side (`0` fills the whole bbox) | yes |
| `pixelate:N` | downsample the document interior so its short side is `N` px, then bilinear back; field text is ~4% of `N`, the face ~35% | yes |
| `downscale:N` | downsample the whole frame to long side `N` px and back | no |

Specs are comma-separated and applied left to right. `mask` and `pixelate` refuse to run if any row in any CSV lacks a bbox, because an untouched row would be un-anonymised. The applied list is recorded under `degrade` in `report.json`.

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
