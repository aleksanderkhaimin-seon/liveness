#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import tensorflow as tf

from eer import compute_eer, get_fr_fa_at_threshold


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


def str2bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_path_remaps(values: list[str]) -> list[tuple[str, str]]:
    remaps = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise argparse.ArgumentTypeError(f"--path-remap expects OLD=NEW, got {item!r}")
            old, new = item.split("=", 1)
            if not old or not new:
                raise argparse.ArgumentTypeError(f"--path-remap expects OLD=NEW, got {item!r}")
            remaps.append((old.rstrip("/"), new.rstrip("/")))
    return remaps


def apply_path_remap(raw_path: str, path_remaps: list[tuple[str, str]]) -> str:
    for old, new in path_remaps:
        if raw_path == old:
            return new
        if raw_path.startswith(old + "/"):
            return new + raw_path[len(old):]
    return raw_path


def normalize_bbox_value(raw_bbox: str, csv_path: Path, row_number: int, require_bbox: bool) -> str:
    bbox = raw_bbox.strip()
    if bbox.lower() in {"", "none", "null", "nan"}:
        return ""

    try:
        values = json.loads(bbox)
    except json.JSONDecodeError as error:
        if not require_bbox:
            return bbox
        raise ValueError(f"Invalid bbox JSON at {csv_path}:{row_number}: {bbox!r}") from error

    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"bbox must be a JSON list [x1,y1,x2,y2] at {csv_path}:{row_number}: {bbox!r}")

    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"bbox values must be numeric at {csv_path}:{row_number}: {bbox!r}") from error

    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"bbox values must be finite at {csv_path}:{row_number}: {bbox!r}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must satisfy x2 > x1 and y2 > y1 at {csv_path}:{row_number}: {bbox!r}")

    return json.dumps([x1, y1, x2, y2], separators=(",", ":"))


def read_csv_dataset(
    csv_path: Path,
    require_bbox: bool = False,
    path_remaps: list[tuple[str, str]] | None = None,
) -> tuple[list[str], list[int], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Expected columns: path,label"
        )

    image_paths = []
    labels = []
    bboxes = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain columns named path and label")
        if require_bbox and "bbox" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a bbox column when --use-bbox-crop is set")

        for row_number, row in enumerate(reader, start=2):
            raw_path = row["path"].strip()
            raw_label = row["label"].strip()
            if not raw_path:
                raise ValueError(f"Empty image path at row {row_number}")

            if path_remaps:
                raw_path = apply_path_remap(raw_path, path_remaps)

            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = csv_path.parent / image_path

            label = int(raw_label)
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1 at row {row_number}, got {raw_label!r}")

            image_paths.append(str(image_path))
            labels.append(label)
            bboxes.append(
                normalize_bbox_value(
                    row.get("bbox", ""),
                    csv_path=csv_path,
                    row_number=row_number,
                    require_bbox=require_bbox,
                )
            )

    if not image_paths:
        raise ValueError(f"No rows found in {csv_path}")

    # A one-class CSV trains to a constant and makes AUC/EER undefined; fail
    # now rather than after several epochs of val_auc=0.5.
    positives = sum(labels)
    negatives = len(labels) - positives
    print(f"{csv_path}: {len(labels)} rows, label 0: {negatives}, label 1: {positives}")
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"{csv_path} contains only label {1 if positives else 0}; "
            "a binary classifier cannot be trained or evaluated on it"
        )
    minority = min(positives, negatives) / len(labels)
    if minority < 0.05:
        print(
            f"WARNING: {csv_path} minority class is {minority:.1%} of rows",
            file=sys.stderr,
        )

    return image_paths, labels, bboxes


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


def parse_bbox(bbox: tf.Tensor) -> tf.Tensor:
    bbox = tf.strings.strip(bbox)
    bbox = tf.strings.regex_replace(bbox, r"^\s*\[", "")
    bbox = tf.strings.regex_replace(bbox, r"\]\s*$", "")
    parts = tf.strings.split(bbox, sep=",")
    values = tf.strings.to_number(parts, out_type=tf.float32)
    return values


def crop_by_bbox(image: tf.Tensor, bbox: tf.Tensor, margin: float) -> tf.Tensor:
    def crop() -> tf.Tensor:
        bbox_values = parse_bbox(bbox)
        bbox_values = tf.ensure_shape(bbox_values, [4])
        image_shape = tf.shape(image)
        image_height = tf.cast(image_shape[0], tf.float32)
        image_width = tf.cast(image_shape[1], tf.float32)

        x1, y1, x2, y2 = tf.unstack(bbox_values, num=4)
        box_width = x2 - x1
        box_height = y2 - y1
        margin_ratio = tf.cast(margin / 100.0, tf.float32)

        x1_margin = x1 - box_width * margin_ratio
        y1_margin = y1 - box_height * margin_ratio
        x2_margin = x2 + box_width * margin_ratio
        y2_margin = y2 + box_height * margin_ratio

        image_height_int = image_shape[0]
        image_width_int = image_shape[1]

        x1_clipped = tf.clip_by_value(x1_margin, 0.0, image_width)
        y1_clipped = tf.clip_by_value(y1_margin, 0.0, image_height)
        x2_clipped = tf.clip_by_value(x2_margin, 0.0, image_width)
        y2_clipped = tf.clip_by_value(y2_margin, 0.0, image_height)

        crop_x = tf.cast(tf.floor(x1_clipped), tf.int32)
        crop_y = tf.cast(tf.floor(y1_clipped), tf.int32)
        crop_x2 = tf.cast(tf.math.ceil(x2_clipped), tf.int32)
        crop_y2 = tf.cast(tf.math.ceil(y2_clipped), tf.int32)

        crop_x = tf.clip_by_value(crop_x, 0, image_width_int - 1)
        crop_y = tf.clip_by_value(crop_y, 0, image_height_int - 1)
        crop_x2 = tf.clip_by_value(crop_x2, crop_x + 1, image_width_int)
        crop_y2 = tf.clip_by_value(crop_y2, crop_y + 1, image_height_int)

        crop_width = crop_x2 - crop_x
        crop_height = crop_y2 - crop_y
        return tf.image.crop_to_bounding_box(image, crop_y, crop_x, crop_height, crop_width)

    has_bbox = tf.greater(tf.strings.length(tf.strings.strip(bbox)), 0)
    return tf.cond(has_bbox, crop, lambda: image)


def load_image(
    path: tf.Tensor,
    label: tf.Tensor,
    bbox: tf.Tensor,
    use_bbox_crop: bool,
    margin: float,
    bbox_aug_prob: float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    if use_bbox_crop:
        image = crop_by_bbox(image, bbox, margin)
    elif bbox_aug_prob > 0.0:
        has_bbox = tf.greater(tf.strings.length(tf.strings.strip(bbox)), 0)
        should_crop = tf.math.logical_and(has_bbox, tf.random.uniform(()) < bbox_aug_prob)
        image = tf.cond(should_crop, lambda: crop_by_bbox(image, bbox, margin), lambda: image)
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


def make_dataset(
    paths: list[str],
    labels: list[int],
    bboxes: list[str],
    batch_size: int,
    training: bool,
    use_bbox_crop: bool,
    margin: float,
    bbox_aug_prob: float = 0.0,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, bboxes))
    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    aug_prob = bbox_aug_prob if training else 0.0
    dataset = dataset.map(
        lambda path, label, bbox: load_image(path, label, bbox, use_bbox_crop, margin, aug_prob),
        num_parallel_calls=AUTOTUNE,
    )
    if training:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def build_learning_rate(
    learning_rate: float,
    use_cosine_decay: bool,
    decay_steps: int,
    min_learning_rate: float,
):
    if not use_cosine_decay:
        return learning_rate

    if decay_steps <= 0:
        raise ValueError("Cosine decay requires decay_steps > 0")
    if min_learning_rate < 0:
        raise ValueError("min_learning_rate must be >= 0")
    if min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate")

    alpha = min_learning_rate / learning_rate if learning_rate > 0 else 0.0
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        alpha=alpha,
    )


def build_model(
    learning_rate,
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


def compute_eer_metrics(labels: list[int], scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float32)

    tar = scores_array[labels_array == 1]
    imp = scores_array[labels_array == 0]
    tar = tar[~np.isnan(tar)]
    imp = imp[~np.isnan(imp)]

    if len(tar) == 0 or len(imp) == 0:
        return {
            "bpcer": float("nan"),
            "apcer": float("nan"),
            "acer": float("nan"),
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "threshold": threshold,
        }

    bpcer, apcer = get_fr_fa_at_threshold(tar=tar, imp=imp, threshold=threshold)
    eer, eer_threshold = compute_eer(tar=tar, imp=imp)
    return {
        "bpcer": float(bpcer),
        "apcer": float(apcer),
        "acer": float((bpcer + apcer) / 2.0),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "threshold": threshold,
    }


class EERCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        bboxes: list[str],
        batch_size: int,
        threshold: float,
        use_bbox_crop: bool,
        margin: float,
        prefix: str = "val",
    ) -> None:
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.bboxes = bboxes
        self.batch_size = batch_size
        self.threshold = threshold
        self.prefix = prefix
        self.dataset = make_dataset(
            paths,
            labels,
            bboxes,
            batch_size=batch_size,
            training=False,
            use_bbox_crop=use_bbox_crop,
            margin=margin,
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}
        logits = self.model.predict(self.dataset, verbose=0).reshape(-1)
        scores = sigmoid_np(logits)
        metrics = compute_eer_metrics(self.labels, scores, self.threshold)

        logs[f"{self.prefix}_bpcer"] = metrics["bpcer"]
        logs[f"{self.prefix}_apcer"] = metrics["apcer"]
        logs[f"{self.prefix}_acer"] = metrics["acer"]
        logs[f"{self.prefix}_eer"] = metrics["eer"]
        logs[f"{self.prefix}_eer_threshold"] = metrics["eer_threshold"]

        print(
            f"\n{self.prefix}_bpcer: {metrics['bpcer']:.4f} - "
            f"{self.prefix}_apcer: {metrics['apcer']:.4f} - "
            f"{self.prefix}_acer: {metrics['acer']:.4f} - "
            f"{self.prefix}_eer: {metrics['eer']:.4f} - "
            f"{self.prefix}_eer_threshold: {metrics['eer_threshold']:.6f}"
        )


class LearningRateLogger(tf.keras.callbacks.Callback):
    def _current_learning_rate(self) -> float:
        learning_rate = self.model.optimizer.learning_rate

        if callable(learning_rate):
            learning_rate = learning_rate(self.model.optimizer.iterations)

        if hasattr(learning_rate, "numpy"):
            return float(learning_rate.numpy())

        return float(tf.keras.backend.get_value(learning_rate))

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}
        logs["learning_rate"] = self._current_learning_rate()
        print(f"\nlearning_rate: {logs['learning_rate']:.10f}")


def write_predictions(
    model: tf.keras.Model,
    paths: list[str],
    labels: list[int],
    bboxes: list[str],
    output_path: Path,
    batch_size: int,
    use_bbox_crop: bool,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = make_dataset(
        paths,
        labels,
        bboxes,
        batch_size=batch_size,
        training=False,
        use_bbox_crop=use_bbox_crop,
        margin=margin,
    )
    logits = model.predict(dataset).reshape(-1)
    scores = sigmoid_np(logits)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label", "logit", "score", "prediction"])
        for path, label, logit, score in zip(paths, labels, logits, scores):
            writer.writerow([path, label, float(logit), float(score), int(logit >= 0.0)])

    return logits, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNetB2 for binary liveness classification.")
    parser.add_argument("--csv", type=Path, default=Path("test_df.csv"), help="CSV with path,label columns.")
    parser.add_argument("--validation-csv", type=Path, help="Optional fixed validation CSV. If omitted, validation is split from --csv.")
    parser.add_argument("--test-csv", type=Path, help="Optional extra CSV used only for final testing and EER.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SM_OUTPUT_DATA_DIR", "runs")) / "efficientnet_b2",
        help="Where to save model and reports.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--cosine-decay", type=str2bool, nargs="?", const=True, default=False, help="Use cosine decay from --learning-rate to --min-learning-rate.")
    parser.add_argument("--min-learning-rate", type=float, default=0.0, help="Final LR floor for cosine decay.")
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-backbone", type=str2bool, nargs="?", const=True, default=False, help="Fine-tune EfficientNetB2 from the first epoch.")
    parser.add_argument("--require-gpu", type=str2bool, nargs="?", const=True, default=False, help="Fail if TensorFlow cannot see a GPU.")
    parser.add_argument("--mixed-precision", type=str2bool, nargs="?", const=True, default=False, help="Use mixed precision, useful on NVIDIA T4.")
    parser.add_argument("--eer-threshold", type=float, default=0.5, help="Threshold for BPCER/APCER/ACER on sigmoid scores.")
    parser.add_argument("--use-bbox-crop", type=str2bool, nargs="?", const=True, default=False, help="Crop images by CSV bbox column before resizing.")
    parser.add_argument("--margin", type=float, default=0.0, help="BBox crop margin percent. 5 expands by 5%%, -5 crops 5%% inside.")
    parser.add_argument("--bbox-aug-prob", type=float, default=0.0, help="Probability of applying bbox crop as augmentation during training (0.0 to disable).")
    parser.add_argument("--comment", type=str, default="", help="Free-text note saved to report.json.")
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        metavar="OLD=NEW[,OLD=NEW]",
        help="Rewrite the CSV path column prefix, e.g. /mnt/dataefs=/opt/ml/input/data/efs. Repeatable or comma separated.",
    )
    parser.add_argument(
        "--export-model-dir",
        type=Path,
        default=Path(os.environ["SM_MODEL_DIR"]) if "SM_MODEL_DIR" in os.environ else None,
        help="Copy best.keras and report.json here after training (SageMaker model artifact dir).",
    )
    args = parser.parse_args()
    args.path_remap = parse_path_remaps(args.path_remap)
    return args


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    configure_runtime(args.require_gpu, args.mixed_precision)

    if args.path_remap:
        print("Path remap:", [f"{old} -> {new}" for old, new in args.path_remap])

    paths, labels, bboxes = read_csv_dataset(
        args.csv,
        require_bbox=args.use_bbox_crop,
        path_remaps=args.path_remap,
    )
    if args.validation_csv:
        train_paths = paths
        train_labels = labels
        train_bboxes = bboxes
        validation_paths, validation_labels, validation_bboxes = read_csv_dataset(
            args.validation_csv,
            require_bbox=args.use_bbox_crop,
            path_remaps=args.path_remap,
        )
        validation_csv = args.validation_csv
    else:
        train_indices, validation_indices = split_indices(labels, args.validation_split, args.seed)
        train_paths = [paths[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        train_bboxes = [bboxes[index] for index in train_indices]
        validation_paths = [paths[index] for index in validation_indices]
        validation_labels = [labels[index] for index in validation_indices]
        validation_bboxes = [bboxes[index] for index in validation_indices]
        validation_csv = None

    train_dataset = make_dataset(
        train_paths,
        train_labels,
        train_bboxes,
        args.batch_size,
        training=True,
        use_bbox_crop=args.use_bbox_crop,
        margin=args.margin,
        bbox_aug_prob=args.bbox_aug_prob,
    )
    validation_dataset = make_dataset(
        validation_paths,
        validation_labels,
        validation_bboxes,
        args.batch_size,
        training=False,
        use_bbox_crop=args.use_bbox_crop,
        margin=args.margin,
    )

    if args.test_csv:
        test_paths, test_labels, test_bboxes = read_csv_dataset(
            args.test_csv,
            require_bbox=args.use_bbox_crop,
            path_remaps=args.path_remap,
        )
        test_csv = args.test_csv
    else:
        test_paths, test_labels = paths, labels
        test_bboxes = bboxes
        test_csv = args.csv

    test_dataset = make_dataset(
        test_paths,
        test_labels,
        test_bboxes,
        args.batch_size,
        training=False,
        use_bbox_crop=args.use_bbox_crop,
        margin=args.margin,
    )

    train_steps_per_epoch = int(np.ceil(len(train_paths) / args.batch_size))
    decay_steps = train_steps_per_epoch * args.epochs
    learning_rate = build_learning_rate(
        learning_rate=args.learning_rate,
        use_cosine_decay=args.cosine_decay,
        decay_steps=decay_steps,
        min_learning_rate=args.min_learning_rate,
    )

    model = build_model(learning_rate, args.train_backbone)
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
        EERCallback(
            paths=validation_paths,
            labels=validation_labels,
            bboxes=validation_bboxes,
            batch_size=args.batch_size,
            threshold=args.eer_threshold,
            use_bbox_crop=args.use_bbox_crop,
            margin=args.margin,
            prefix="val",
        ),
        LearningRateLogger(),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(args.output_dir / "tensorboard"),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
        tf.keras.callbacks.CSVLogger(str(args.output_dir / "history.csv")),
    ]

    print(f"Starting training for requested epochs: {args.epochs}")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight(train_labels),
    )

    model.save(args.output_dir / "last.keras")
    model = tf.keras.models.load_model(args.output_dir / "best.keras", compile=True)
    metrics = model.evaluate(test_dataset, return_dict=True)
    _, test_scores = write_predictions(
        model,
        test_paths,
        test_labels,
        test_bboxes,
        args.output_dir / "test_predictions.csv",
        args.batch_size,
        use_bbox_crop=args.use_bbox_crop,
        margin=args.margin,
    )
    eer_metrics = compute_eer_metrics(test_labels, test_scores, args.eer_threshold)

    report = {
        "csv": str(args.csv),
        "validation_csv": str(validation_csv) if validation_csv else None,
        "test_csv": str(test_csv),
        "samples": len(paths),
        "test_samples": len(test_paths),
        "train_samples": len(train_paths),
        "validation_samples": len(validation_paths),
        "requested_epochs": args.epochs,
        "completed_epochs": len(history.history.get("loss", [])),
        "learning_rate": {
            "initial": args.learning_rate,
            "cosine_decay": args.cosine_decay,
            "min": args.min_learning_rate,
            "decay_steps": decay_steps if args.cosine_decay else None,
            "train_steps_per_epoch": train_steps_per_epoch,
        },
        "bbox_crop": {
            "enabled": args.use_bbox_crop,
            "margin": args.margin,
            "aug_prob": args.bbox_aug_prob,
        },
        "comment": args.comment,
        "test_metrics": metrics,
        "test_eer_metrics": eer_metrics,
        "history": history.history,
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    if args.export_model_dir:
        args.export_model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("best.keras", "last.keras", "report.json"):
            source = args.output_dir / name
            if source.exists():
                shutil.copy2(source, args.export_model_dir / name)
        print(f"Exported model artifacts to {args.export_model_dir}")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "test_metrics": metrics,
        "test_eer_metrics": eer_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
