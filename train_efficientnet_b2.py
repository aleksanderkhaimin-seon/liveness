#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import albumentations as A
import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 512


AUGMENTER = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.92, 1.08),
            rotate=(-10.8, 10.8),
            border_mode=1,
            p=1.0,
        ),
        A.MultiplicativeNoise(
            multiplier=(0.8, 1.2),
            per_channel=False,
            p=1.0,
        ),
    ]
)


def configure_runtime(require_gpu: bool, mixed_precision: bool) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow GPUs:", [gpu.name for gpu in gpus])
    else:
        print("TensorFlow GPUs: none")
        if require_gpu:
            raise RuntimeError(
                "TensorFlow cannot see a GPU. Check Docker GPU passthrough with "
                "`docker compose exec liveness-training nvidia-smi`."
            )

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: enabled")


def read_csv_dataset(csv_path: Path) -> tuple[list[str], list[int]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Expected columns: path,label"
        )

    image_paths = []
    labels = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain columns named path and label")

        for row_number, row in enumerate(reader, start=2):
            raw_path = row["path"].strip()
            raw_label = row["label"].strip()
            if not raw_path:
                raise ValueError(f"Empty image path at row {row_number}")

            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = csv_path.parent / image_path

            label = int(raw_label)
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1 at row {row_number}, got {raw_label!r}")

            image_paths.append(str(image_path))
            labels.append(label)

    if not image_paths:
        raise ValueError(f"No rows found in {csv_path}")

    return image_paths, labels


def split_indices(labels: list[int], validation_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels, dtype=np.int32)
    train_indices = []
    validation_indices = []

    for label in (0, 1):
        class_indices = np.where(labels_array == label)[0]
        rng.shuffle(class_indices)

        validation_count = max(1, int(round(len(class_indices) * validation_split)))
        if validation_count >= len(class_indices):
            validation_count = max(0, len(class_indices) - 1)

        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    if not train_indices or not validation_indices:
        raise ValueError("Could not create a non-empty train/validation split")

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return np.asarray(train_indices), np.asarray(validation_indices)


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def augment_image_np(image: np.ndarray) -> np.ndarray:
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    augmented = AUGMENTER(image=image_uint8)["image"]
    return augmented.astype(np.float32)


def augment_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.numpy_function(augment_image_np, [image], tf.float32)
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    return image, label


def make_dataset(paths: list[str], labels: list[int], batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def build_model(
    learning_rate: float,
    train_backbone: bool,
    backbone_weights: str | None = "imagenet",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    backbone = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=backbone_weights,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = train_backbone

    x = backbone(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32", name="live_score")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
            tf.keras.metrics.AUC(name="auc", from_logits=True),
            tf.keras.metrics.Precision(name="precision", thresholds=0.0),
            tf.keras.metrics.Recall(name="recall", thresholds=0.0),
        ],
    )
    return model


def class_weight(labels: list[int]) -> dict[int, float]:
    counts = np.bincount(np.asarray(labels, dtype=np.int32), minlength=2)
    total = counts.sum()
    return {
        class_id: float(total / (2 * count)) if count > 0 else 1.0
        for class_id, count in enumerate(counts)
    }


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def write_predictions(model: tf.keras.Model, paths: list[str], labels: list[int], output_path: Path, batch_size: int) -> None:
    dataset = make_dataset(paths, labels, batch_size=batch_size, training=False)
    logits = model.predict(dataset).reshape(-1)
    scores = sigmoid_np(logits)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label", "logit", "score", "prediction"])
        for path, label, logit, score in zip(paths, labels, logits, scores):
            writer.writerow([path, label, float(logit), float(score), int(logit >= 0.0)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNetB2 for binary liveness classification.")
    parser.add_argument("--csv", type=Path, default=Path("test_df.csv"), help="CSV with path,label columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/efficientnet_b2"), help="Where to save model and reports.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-backbone", action="store_true", help="Fine-tune EfficientNetB2 from the first epoch.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if TensorFlow cannot see a GPU.")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision, useful on NVIDIA T4.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    configure_runtime(args.require_gpu, args.mixed_precision)

    paths, labels = read_csv_dataset(args.csv)
    train_indices, validation_indices = split_indices(labels, args.validation_split, args.seed)

    train_paths = [paths[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    validation_paths = [paths[index] for index in validation_indices]
    validation_labels = [labels[index] for index in validation_indices]

    train_dataset = make_dataset(train_paths, train_labels, args.batch_size, training=True)
    validation_dataset = make_dataset(validation_paths, validation_labels, args.batch_size, training=False)
    test_dataset = make_dataset(paths, labels, args.batch_size, training=False)

    model = build_model(args.learning_rate, args.train_backbone)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "checkpoints" / "epoch_{epoch:03d}_val_auc_{val_auc:.4f}.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=False,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(args.output_dir / "tensorboard"),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(args.output_dir / "history.csv")),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight(train_labels),
    )

    model.save(args.output_dir / "last.keras")
    metrics = model.evaluate(test_dataset, return_dict=True)
    write_predictions(model, paths, labels, args.output_dir / "test_predictions.csv", args.batch_size)

    report = {
        "csv": str(args.csv),
        "samples": len(paths),
        "train_samples": len(train_paths),
        "validation_samples": len(validation_paths),
        "test_metrics": metrics,
        "history": history.history,
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps({"output_dir": str(args.output_dir), "test_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
