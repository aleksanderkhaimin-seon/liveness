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

## Useful Options

```bash
python train_efficientnet_b2.py \
  --csv test_df.csv \
  --output-dir runs/efficientnet_b2 \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --validation-split 0.2
```

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
