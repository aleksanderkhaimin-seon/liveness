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

`best.keras` (and therefore the test metrics, which are computed from it) is selected on `val_eer` (lower is better). `--checkpoint-monitor val_auc` restores the previous behaviour; `report.json` records the monitor and the selected epoch under `checkpoint`. Validation is production data and EER is the metric acted on; in the September 2026 runs `val_auc` peaked at epoch 0 while `val_eer` often did not, so the two monitors select different models.

With `--cosine-decay`, the optimizer uses cosine decay from `--learning-rate` down to `--min-learning-rate` across the requested number of epochs. The effective `learning_rate` is logged at the end of each epoch to `history.csv` and TensorBoard.

With `--use-bbox-crop`, CSV files must include a `bbox` column formatted like `[x1,y1,x2,y2]`. Cropping is applied before resize for train, validation, and test datasets. `--margin 5` expands the crop by 5% of bbox width/height on every side; `--margin -5` crops 5% inside the bbox.

Fine-tune the EfficientNetB2 backbone immediately:

```bash
python train_efficientnet_b2.py --csv test_df.csv --train-backbone
```

## Input Size And Resize Mode

`--image-size N` (config key `image_size`, default 512) sets the network input side; `--resize-mode squash|letterbox` (default `squash`) sets how a frame is brought to that square. `squash` is the historical bilinear resize ignoring aspect. `letterbox` downsamples with an area filter only if the frame is larger than the target, never upsamples, and pads centred with the frame's mean colour — a frame that already fits is copied pixel for pixel. Frames exported with `export_lowres_frames.py --mode doc --size 96` are 164–211 px on the long side, so at `--image-size 224 --resize-mode letterbox` almost none of them is resampled at all.

`report.json` records `input.{image_size, resize_mode}` and `throughput.{train_seconds_per_epoch, train_images_per_sec}`. `predict_onnx_csv.py`, `predict_checkpoint_csv.py`, `convert_checkpoint_to_onnx.py` and `infer_onnx.py` still assume 512 and need the size passed through before a model trained at another size is exported.

## Anonymisation Degradations

`--degrade` applies content-destroying transforms to the decoded frame before any bbox crop and before the resize to 512. They run identically on train, validation and test and on both classes, so they cannot become a label shortcut. Use them to measure how much EER survives a given anonymisation before building the export for it.

```bash
python train_efficientnet_b2.py --csv train.csv --degrade mask:0.05
python train_efficientnet_b2.py --csv train.csv --degrade mask:0.05,downscale:192
DEGRADE=mask:0.05 docker compose up liveness-train
python sagemaker_job/launch.py --config configs/train-gpu.json --degrade mask:0.05
```

| spec | effect | needs bbox |
|---|---|---|
| `mask:BAND` | fill the document interior with its per-channel mean, keeping a border band `BAND` × bbox size on each side (`0` fills the whole bbox) | yes |
| `pixelate:N` | downsample the document interior so its short side is `N` px, then bilinear back; field text is ~4% of `N`, the face ~35% | yes |
| `downscale:N` | downsample the whole frame to long side `N` px and back | no |
| `downscale_doc:N` | downsample the whole frame so the *document* long side is `N` px, and back — a per-image legibility bound (`doc:96` ≈ `frame:192` for the median capture, harsher for close-ups) | yes |

Results so far (`runs/train_reports`, 3 epochs, production validation = `ProdTest-0.3`): `downscale:192` matched the un-degraded baseline on both Pinterest test (3.0 vs 3.2% EER) and production (15.9 vs 15.0%); native-resolution patches reached 1.7% per document on Pinterest but 38% on production. Coarse cues transfer, fine texture does not.

To materialise the low-resolution representation as files rather than a training-time flag, use `export_lowres_frames.py`; `--absolute-paths` is needed when the resulting `frames.csv` goes through `sagemaker_job/launch.py`:

```bash
python export_lowres_frames.py data/train.csv --out-dir /home/sagemaker-user/seon-data-efs/data/anon_lowres/train --mode doc --size 96 --absolute-paths --workers 16
```

Specs are comma-separated and applied left to right. `mask` and `pixelate` refuse to run if any row in any CSV lacks a bbox, because an untouched row would be un-anonymised. The applied list is recorded under `degrade` in `report.json`.

## Augmentations

Training uses `albumentations` in the `tf.data` input pipeline:

- horizontal flip with probability `0.5`
- affine rotation from `-10.8` to `10.8` degrees
- affine scale from `0.92` to `1.08`
- brightness multiplier from `0.8` to `1.2`

Validation, testing, checkpoint conversion, and ONNX inference do not include augmentations.

### Frequency augmentations

In the training manifests the label tracks the source. Lives are mostly 3024×4032 phone photos and scraped web images. Attacks are mostly 1920×1080 replay frames. In production both classes arrive at about 1920 px through the same pipeline. A model can therefore separate the training classes by their capture fingerprint instead of the attack: the resampling ratio to the network input, JPEG history, sensor noise and sharpening. These cues all sit in the image spectrum. `--freq-aug` (config key `freq_aug`) adds training-only transforms that make those cues unreliable:

| spec | effect |
|---|---|
| `rescale:P` | before the network resize, area-downsample the frame to a long side log-uniform in [S, 4S] (S = `image_size`; never upsampled), then re-encode it as JPEG at quality 60–95 |
| `bandstop:P` | attenuate a random ring of the network input's spectrum (centre 0.1–1.0 of Nyquist, width 0.05–0.3, depth 50–100%) |
| `ampmix:P[:ETA]` | per batch, mix each image's Fourier amplitude with that of a random image of the other class and keep its own phase; mixing weight U(0, ETA), ETA defaults to 0.5 |

```bash
python sagemaker_job/launch.py --config configs/train-gpu.json --freq-aug rescale:0.5,bandstop:0.3,ampmix:0.5
```

They are recorded under `freq_aug` in `report.json`. On a laptop CPU they add about 2 ms per 512 px image, which is small next to decoding 4K JPEGs.

### Orientation

Most training attacks are landscape frames and most training lives are portrait; in production both are mostly portrait. After the squash to a square, frame orientation becomes a horizontal-vs-vertical frequency imbalance. The September 30 model (`runs/freq/1833`) used it as an attack cue: on production, 62% of landscape lives and 4% of landscape attacks were misclassified, against 11% and 14% in portrait. `--rot90-prob P` (config key `rot90_prob`) rotates each training frame a quarter turn, clockwise or counter-clockwise, with probability `P`. This happens after the bbox crop, so orientation no longer tracks the label. It is recorded as `rot90_prob` in `report.json`.

```bash
python sagemaker_job/launch.py --config configs/train-gpu.json --rot90-prob 0.5 --freq-aug ampmix:0.5
```

To check whether an existing model depends on orientation without retraining, re-score with every frame rotated. `predict_checkpoint_csv.py` and `predict_onnx_csv.py` take `--rotate 0|90|180|270` (counter-clockwise, after the bbox crop). If the model relies on orientation, portrait attacks should become easier to catch and portrait lives should start failing.

```bash
python predict_checkpoint_csv.py data/ProdTest-0.3.csv --checkpoint runs/x/best.keras --output-csv runs/x/prod_rot90.csv --rotate 90
```

`frequency_shortcuts.py` checks whether such a shortcut exists and whether a model uses it. It measures the network input exactly as validation sees it. A spectrum-only linear probe is fitted on train and scored on prod. A per-band, per-orientation effect size (attack − live) is computed for train and prod. With `--checkpoint`, it also reports the EER change on each split when one radial band is removed. A band that matters on the in-distribution test but not on prod is a band the model relies on that does not transfer.

```bash
python frequency_shortcuts.py --train data/train.csv --prod data/ProdTest-0.3.csv \
  --test data/test_Pinterest_v_1_checked.csv --checkpoint eval_results/<job>/output/best.keras \
  --out-dir runs/freq/<job>
```

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
