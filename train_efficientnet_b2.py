#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import tensorflow as tf

from eer import compute_eer, get_fr_fa_at_threshold


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 512          # set from --image-size / config in main()
RESIZE_MODE = "squash"    # set from --resize-mode / config in main(): squash | letterbox
RESIZE_MODES = ("squash", "letterbox")


def set_input_geometry(image_size: int, resize_mode: str) -> None:
    global IMAGE_SIZE, RESIZE_MODE
    if image_size < 64:
        raise ValueError(f"image_size must be at least 64, got {image_size}")
    if resize_mode not in RESIZE_MODES:
        raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {resize_mode!r}")
    IMAGE_SIZE = int(image_size)
    RESIZE_MODE = resize_mode


def fit_to_input(image: tf.Tensor) -> tf.Tensor:
    """Bring a decoded float image to (IMAGE_SIZE, IMAGE_SIZE, 3).

    squash    : bilinear resize to the square, ignoring aspect (historical behaviour).
    letterbox : downscale with an area filter only if the frame exceeds the target,
                never upscale, then pad centred with the frame's per-channel mean.
                A frame that already fits is copied pixel for pixel.
    """
    if RESIZE_MODE == "squash":
        return tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")

    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    scale = tf.minimum(1.0, IMAGE_SIZE / tf.cast(tf.maximum(height, width), tf.float32))

    def shrink() -> tf.Tensor:
        new_height = tf.maximum(1, tf.cast(tf.round(tf.cast(height, tf.float32) * scale), tf.int32))
        new_width = tf.maximum(1, tf.cast(tf.round(tf.cast(width, tf.float32) * scale), tf.int32))
        return tf.image.resize(image, (new_height, new_width), method="area")

    image = tf.cond(scale < 1.0, shrink, lambda: image)
    shape = tf.shape(image)
    height = tf.minimum(shape[0], IMAGE_SIZE)
    width = tf.minimum(shape[1], IMAGE_SIZE)
    image = image[:height, :width]
    top = (IMAGE_SIZE - height) // 2
    left = (IMAGE_SIZE - width) // 2
    paddings = [[top, IMAGE_SIZE - height - top], [left, IMAGE_SIZE - width - left], [0, 0]]
    mean = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    # Pad with zeros and add the mean only where padding was inserted, so the
    # frame itself is copied exactly rather than passing through a subtract/add.
    pad_mask = tf.pad(tf.zeros_like(image[..., :1]), paddings, constant_values=1.0)
    return tf.pad(image, paddings) + pad_mask * mean


_BASE_AUGMENTATIONS = [
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

# Extra photometric / camera / framing variation aimed at the train->production
# gap: it is coarse capture conditions (display brightness and colour cast, glare,
# sensor noise, compression, tilt) that differ between collection setups and
# production, not fine texture. Applied after the resize, so blur and noise act
# at network-input scale. Only albumentations arguments that are stable across
# 1.x and 2.x are used.
_DOMAIN_AUGMENTATIONS = [
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 140), p=1.0),
        ],
        p=0.8,
    ),
    A.OneOf(
        [
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ],
        p=0.6,
    ),
    A.OneOf([A.GaussianBlur(p=1.0), A.MotionBlur(p=1.0), A.Defocus(p=1.0)], p=0.3),
    A.GaussNoise(p=0.3),
    A.ImageCompression(p=0.4),
    A.Perspective(scale=(0.02, 0.08), p=0.3),
    A.RandomShadow(p=0.2),
]

AUGMENT_PRESETS = {
    "base": _BASE_AUGMENTATIONS,
    "domain": _BASE_AUGMENTATIONS + _DOMAIN_AUGMENTATIONS,
}
AUGMENTER = A.Compose(AUGMENT_PRESETS["base"])


def set_augment_preset(name: str) -> None:
    global AUGMENTER
    if name not in AUGMENT_PRESETS:
        raise ValueError(f"augment must be one of {sorted(AUGMENT_PRESETS)}, got {name!r}")
    AUGMENTER = A.Compose(AUGMENT_PRESETS[name])


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
    with_dataset: bool = False,
) -> tuple[list[str], list[int], list[str]] | tuple[list[str], list[int], list[str], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Expected columns: path,label"
        )

    image_paths = []
    labels = []
    bboxes = []
    datasets = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain columns named path and label")
        if require_bbox and "bbox" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a bbox column when --use-bbox-crop is set")
        if with_dataset and "dataset" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must contain a dataset column when cover_attack_datasets is set"
            )

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
            bboxes.append(
                normalize_bbox_value(
                    row.get("bbox", ""),
                    csv_path=csv_path,
                    row_number=row_number,
                    require_bbox=require_bbox,
                )
            )
            if with_dataset:
                datasets.append((row.get("dataset") or "").strip() or "unknown")

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

    if with_dataset:
        return image_paths, labels, bboxes, datasets
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


# -- anonymisation degradations ---------------------------------------------
#
# Applied to the decoded full frame, in original pixel coordinates, before any
# bbox crop and before the resize to IMAGE_SIZE. They run on every split and
# both classes identically, so they cannot become a label shortcut. Set once
# from --degrade in main(); module-level like AUGMENTER.
#
#   mask:BAND        fill the document interior with its own per-channel mean,
#                    leaving a border band of BAND x bbox size (0 = whole bbox)
#   pixelate:N       downsample the document interior so its short side is N px,
#                    then bilinear-upsample back (text ~4% of N, face ~35% of N)
#   downscale:N      downsample the whole frame to long side N px and back
#   downscale_doc:N  downsample the whole frame so the DOCUMENT long side is N px,
#                    and back -- a per-image legibility bound rather than a
#                    population statistic (a close-up is scaled harder)
#
# Example: --degrade mask:0.05,downscale:192

DEGRADE_KINDS = ("mask", "pixelate", "downscale", "downscale_doc")
DEGRADATIONS: list[tuple[str, float]] = []


def parse_degrade(text: str) -> list[tuple[str, float]]:
    specs: list[tuple[str, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        kind, _, value = item.partition(":")
        kind = kind.strip().lower()
        if kind not in DEGRADE_KINDS:
            raise ValueError(f"Unknown degradation {kind!r}; expected one of {DEGRADE_KINDS}")
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError(f"Degradation {item!r} needs a numeric value, e.g. mask:0.05") from error
        if kind == "mask" and not 0.0 <= number < 0.5:
            raise ValueError(f"mask band must be in [0, 0.5), got {number}")
        if kind in ("pixelate", "downscale", "downscale_doc") and number < 8:
            raise ValueError(f"{kind} target must be at least 8 px, got {number}")
        specs.append((kind, number))
    return specs


def bbox_rect(image: tf.Tensor, bbox: tf.Tensor, shrink: float) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Integer [x1, y1, x2, y2] of the bbox shrunk by `shrink` x its size per side, clipped to the image."""
    values = tf.ensure_shape(parse_bbox(bbox), [4])
    x1, y1, x2, y2 = tf.unstack(values, num=4)
    box_width = x2 - x1
    box_height = y2 - y1
    x1 = x1 + box_width * shrink
    x2 = x2 - box_width * shrink
    y1 = y1 + box_height * shrink
    y2 = y2 - box_height * shrink
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    x1 = tf.cast(tf.floor(tf.clip_by_value(x1, 0.0, width)), tf.int32)
    y1 = tf.cast(tf.floor(tf.clip_by_value(y1, 0.0, height)), tf.int32)
    x2 = tf.cast(tf.math.ceil(tf.clip_by_value(x2, 0.0, width)), tf.int32)
    y2 = tf.cast(tf.math.ceil(tf.clip_by_value(y2, 0.0, height)), tf.int32)
    return x1, y1, x2, y2


def rect_mask(image: tf.Tensor, x1: tf.Tensor, y1: tf.Tensor, x2: tf.Tensor, y2: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(image)
    yy = tf.range(shape[0])[:, tf.newaxis]
    xx = tf.range(shape[1])[tf.newaxis, :]
    inside = (yy >= y1) & (yy < y2) & (xx >= x1) & (xx < x2)
    return inside[..., tf.newaxis]


def degrade_mask(image: tf.Tensor, bbox: tf.Tensor, band: float) -> tf.Tensor:
    x1, y1, x2, y2 = bbox_rect(image, bbox, band)

    def apply() -> tf.Tensor:
        region = image[y1:y2, x1:x2]
        fill = tf.reduce_mean(region, axis=[0, 1], keepdims=True)
        return tf.where(rect_mask(image, x1, y1, x2, y2), tf.broadcast_to(fill, tf.shape(image)), image)

    return tf.cond((x2 > x1) & (y2 > y1), apply, lambda: image)


def degrade_pixelate(image: tf.Tensor, bbox: tf.Tensor, target_short: float) -> tf.Tensor:
    x1, y1, x2, y2 = bbox_rect(image, bbox, 0.0)
    region_height = y2 - y1
    region_width = x2 - x1
    short = tf.cast(tf.minimum(region_height, region_width), tf.float32)
    factor = target_short / tf.maximum(short, 1.0)

    def apply() -> tf.Tensor:
        region = image[y1:y2, x1:x2]
        small_height = tf.maximum(1, tf.cast(tf.round(tf.cast(region_height, tf.float32) * factor), tf.int32))
        small_width = tf.maximum(1, tf.cast(tf.round(tf.cast(region_width, tf.float32) * factor), tf.int32))
        small = tf.image.resize(region, (small_height, small_width), method="area")
        back = tf.image.resize(small, (region_height, region_width), method="bilinear")
        shape = tf.shape(image)
        padded = tf.pad(back, [[y1, shape[0] - y2], [x1, shape[1] - x2], [0, 0]])
        return tf.where(rect_mask(image, x1, y1, x2, y2), padded, image)

    return tf.cond((x2 > x1) & (y2 > y1) & (factor < 1.0), apply, lambda: image)


def degrade_downscale(image: tf.Tensor, long_side: float) -> tf.Tensor:
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    factor = long_side / tf.cast(tf.maximum(height, width), tf.float32)

    def apply() -> tf.Tensor:
        small_height = tf.maximum(1, tf.cast(tf.round(tf.cast(height, tf.float32) * factor), tf.int32))
        small_width = tf.maximum(1, tf.cast(tf.round(tf.cast(width, tf.float32) * factor), tf.int32))
        small = tf.image.resize(image, (small_height, small_width), method="area")
        return tf.image.resize(small, (height, width), method="bilinear")

    return tf.cond(factor < 1.0, apply, lambda: image)


def degrade_downscale_doc(image: tf.Tensor, bbox: tf.Tensor, doc_long_side: float) -> tf.Tensor:
    """Downscale the whole frame so the document's long side becomes doc_long_side px."""
    x1, y1, x2, y2 = bbox_rect(image, bbox, 0.0)
    doc_long = tf.cast(tf.maximum(x2 - x1, y2 - y1), tf.float32)
    shape = tf.shape(image)
    frame_long = tf.cast(tf.maximum(shape[0], shape[1]), tf.float32)
    # Express the target as a frame long side so the frame-level helper does the work.
    target_frame_long = frame_long * doc_long_side / tf.maximum(doc_long, 1.0)
    return tf.cond(doc_long > doc_long_side, lambda: degrade_downscale(image, target_frame_long), lambda: image)


def apply_degradations(image: tf.Tensor, bbox: tf.Tensor) -> tf.Tensor:
    if not DEGRADATIONS:
        return image
    has_bbox = tf.greater(tf.strings.length(tf.strings.strip(bbox)), 0)
    for kind, value in DEGRADATIONS:
        if kind == "mask":
            image = tf.cond(has_bbox, lambda img=image, v=value: degrade_mask(img, bbox, v), lambda img=image: img)
        elif kind == "pixelate":
            image = tf.cond(has_bbox, lambda img=image, v=value: degrade_pixelate(img, bbox, v), lambda img=image: img)
        elif kind == "downscale":
            image = degrade_downscale(image, value)
        elif kind == "downscale_doc":
            image = tf.cond(has_bbox, lambda img=image, v=value: degrade_downscale_doc(img, bbox, v), lambda img=image: img)
    return image


# -- frequency-shortcut augmentations (training only) --------------------------
#
# In the training manifests the label is the source: lives are phone photos
# (mostly 3024x4032) and scraped web images, attacks are replay captures
# (mostly 1920x1080 frames). Whatever fingerprints the capture pipeline --
# the resampling ratio down to the network input, JPEG history, sensor noise
# and sharpening, the overall spectral envelope -- separates the classes in
# training and means nothing in production, where both classes arrive through
# the same ~1920 px pipeline. These transforms make such cues unreliable. They
# never run on validation, test or inference. Set once from --freq-aug.
#
#   rescale:P        area-downsample the (cropped) frame so its long side is
#                    log-uniform in [IMAGE_SIZE, 4 x IMAGE_SIZE] (never upsample),
#                    then re-encode as JPEG at quality 60-95. The final resize to
#                    the network input then runs at a ratio unrelated to the
#                    source resolution.
#   bandstop:P       attenuate a random ring of the network input's spectrum
#                    (centre 0.1-1.0 of Nyquist, width 0.05-0.3, depth 50-100%)
#                    so that no single band can carry the decision.
#   ampmix:P[:ETA]   per batch: mix each image's Fourier amplitude with that of
#                    a random image of the other class, keeping its own phase:
#                    amplitude = (1 - l) own + l other, l ~ U(0, ETA), ETA 0.5 by
#                    default. The spectral envelope stops being a class property;
#                    structure (edges, moire geometry, bezels) lives in the phase.
#
# Example: --freq-aug rescale:0.5,bandstop:0.3,ampmix:0.5

FREQ_AUG_KINDS = ("rescale", "bandstop", "ampmix")
FREQ_AUG: dict[str, tuple[float, float]] = {}


def parse_freq_aug(text: str) -> dict[str, tuple[float, float]]:
    specs: dict[str, tuple[float, float]] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        kind, *values = [part.strip() for part in item.split(":")]
        kind = kind.lower()
        if kind not in FREQ_AUG_KINDS:
            raise ValueError(f"Unknown frequency augmentation {kind!r}; expected one of {FREQ_AUG_KINDS}")
        if kind in specs:
            raise ValueError(f"{kind} given twice")
        if not values or len(values) > (2 if kind == "ampmix" else 1):
            raise ValueError(f"{item!r}: expected {kind}:P" + ("[:ETA]" if kind == "ampmix" else ""))
        try:
            numbers = [float(value) for value in values]
        except ValueError as error:
            raise ValueError(f"{item!r} needs numeric values, e.g. {kind}:0.5") from error
        prob = numbers[0]
        if not 0.0 < prob <= 1.0:
            raise ValueError(f"{kind} probability must be in (0, 1], got {prob}")
        param = numbers[1] if len(numbers) > 1 else 0.5
        if kind == "ampmix" and not 0.0 < param <= 1.0:
            raise ValueError(f"ampmix ETA must be in (0, 1], got {param}")
        specs[kind] = (prob, param)
    return specs


def random_rescale(image: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    target = tf.exp(tf.random.uniform((), math.log(IMAGE_SIZE), math.log(4 * IMAGE_SIZE)))
    scale = target / tf.maximum(height, width)

    def shrink() -> tf.Tensor:
        size = tf.maximum(1, tf.cast(tf.round(tf.stack([height, width]) * scale), tf.int32))
        return tf.image.resize(image, size, method="area")

    resized = tf.cond(scale < 1.0, shrink, lambda: tf.cast(image, tf.float32))
    resized = tf.cast(tf.clip_by_value(tf.round(resized), 0.0, 255.0), tf.uint8)
    quality = tf.random.uniform((), 60, 96, dtype=tf.int32)
    recoded = tf.image.adjust_jpeg_quality(resized, quality)
    recoded = tf.ensure_shape(recoded, [None, None, 3])
    return tf.cast(recoded, image.dtype)


def radial_frequency(size: int) -> np.ndarray:
    """|f| / Nyquist on the rfft2d grid of a size x size image, shape (size, size // 2 + 1)."""
    fy = np.fft.fftfreq(size)[:, np.newaxis]
    fx = np.fft.rfftfreq(size)[np.newaxis, :]
    return (np.sqrt(fy**2 + fx**2) / 0.5).astype(np.float32)


def apply_spectral_gain(images: tf.Tensor, gain: tf.Tensor) -> tf.Tensor:
    """Multiply the 2-D spectrum of (B, H, W, C) images by a real gain on the rfft2d grid.

    `gain` broadcasts against (B, C, H, W // 2 + 1). Output is clipped to [0, 255].
    """
    channels_first = tf.transpose(tf.cast(images, tf.float32), [0, 3, 1, 2])
    spectrum = tf.signal.rfft2d(channels_first) * tf.cast(gain, tf.complex64)
    filtered = tf.signal.irfft2d(spectrum, fft_length=tf.shape(channels_first)[2:])
    return tf.clip_by_value(tf.transpose(filtered, [0, 2, 3, 1]), 0.0, 255.0)


def random_bandstop(image: tf.Tensor) -> tf.Tensor:
    rho = tf.constant(radial_frequency(IMAGE_SIZE))
    centre = tf.random.uniform((), 0.1, 1.0)
    half_width = tf.random.uniform((), 0.025, 0.15)
    depth = tf.random.uniform((), 0.5, 1.0)
    # Raised-cosine notch: full depth at the centre, back to 1 at centre +- half_width.
    t = tf.clip_by_value((rho - centre) / half_width, -1.0, 1.0)
    gain = 1.0 - depth * 0.5 * (1.0 + tf.cos(math.pi * t))
    return apply_spectral_gain(image[tf.newaxis], gain)[0]


def opposite_class_partner(labels: tf.Tensor) -> tf.Tensor:
    """For each batch row, a random row of the other label (any other row if the batch has one label)."""
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    batch = tf.shape(labels)[0]
    same = tf.equal(labels[:, tf.newaxis], labels[tf.newaxis, :])
    noise = tf.random.uniform([batch, batch])
    opposite = tf.argmax(tf.where(same, -1.0, noise), axis=1, output_type=tf.int32)
    other = tf.argmax(tf.where(tf.eye(batch, dtype=tf.bool), -1.0, noise), axis=1, output_type=tf.int32)
    return tf.where(tf.reduce_any(~same, axis=1), opposite, other)


def amplitude_mix(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    prob, eta = FREQ_AUG["ampmix"]
    batch = tf.shape(images)[0]
    channels_first = tf.transpose(images, [0, 3, 1, 2])
    spectrum = tf.signal.rfft2d(channels_first)
    amplitude = tf.abs(spectrum)
    applied = tf.cast(tf.random.uniform([batch, 1, 1, 1]) < prob, tf.float32)
    weight = tf.random.uniform([batch, 1, 1, 1], 0.0, eta) * applied
    mixed = (1.0 - weight) * amplitude + weight * tf.gather(amplitude, opposite_class_partner(labels))
    spectrum = spectrum * tf.cast(tf.math.divide_no_nan(mixed, amplitude), tf.complex64)
    out = tf.signal.irfft2d(spectrum, fft_length=tf.shape(channels_first)[2:])
    return tf.clip_by_value(tf.transpose(out, [0, 2, 3, 1]), 0.0, 255.0), labels


def load_image(
    path: tf.Tensor,
    label: tf.Tensor,
    bbox: tf.Tensor,
    use_bbox_crop: bool,
    margin: float,
    bbox_aug_prob: float = 0.0,
    training: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    if DEGRADATIONS:
        # Degradations work in float; without them the frame stays uint8 through
        # crop and resize (tf.image.resize converts internally), which avoids
        # materialising a 4x larger full-resolution tensor per image.
        image = tf.cast(image, tf.float32)
        image = apply_degradations(image, bbox)
    if use_bbox_crop:
        image = crop_by_bbox(image, bbox, margin)
    elif bbox_aug_prob > 0.0:
        has_bbox = tf.greater(tf.strings.length(tf.strings.strip(bbox)), 0)
        should_crop = tf.math.logical_and(has_bbox, tf.random.uniform(()) < bbox_aug_prob)
        image = tf.cond(should_crop, lambda: crop_by_bbox(image, bbox, margin), lambda: image)
    if training and "rescale" in FREQ_AUG:
        prob = FREQ_AUG["rescale"][0]
        image = tf.cond(tf.random.uniform(()) < prob, lambda: random_rescale(image), lambda: image)
    image = fit_to_input(image)
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
    if "bandstop" in FREQ_AUG:
        prob = FREQ_AUG["bandstop"][0]
        image = tf.cond(tf.random.uniform(()) < prob, lambda: random_bandstop(image), lambda: image)
    return image, label


def batch_training(dataset: tf.data.Dataset, batch_size: int, drop_remainder: bool) -> tf.data.Dataset:
    """Per-image augmentation, batching, then the batch-level amplitude mix."""
    dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if "ampmix" in FREQ_AUG:
        dataset = dataset.map(amplitude_mix, num_parallel_calls=AUTOTUNE)
    return dataset.prefetch(AUTOTUNE)


def plan_attack_coverage_batches(
    labels: list[int],
    datasets: list[str],
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Row indices of shape (n_batches, batch_size).

    Every batch contains one label-1 row from each dataset that has attacks.
    Remaining slots stay half live and half attack when that many attack
    datasets fit in half the batch. The tail that does not fill a batch is
    dropped so every step has the full set of attack datasets.
    """
    if batch_size < 2:
        raise ValueError(f"batch_size must be at least 2, got {batch_size}")
    labels_array = np.asarray(labels, dtype=np.int32)
    dataset_array = np.asarray(datasets)
    if len(labels_array) != len(dataset_array):
        raise ValueError("labels and datasets must have the same length")

    attack_names = sorted(
        {str(name) for name, label in zip(dataset_array, labels_array) if int(label) == 1}
    )
    if not attack_names:
        raise ValueError("cover_attack_datasets needs at least one label-1 row")
    if len(attack_names) > batch_size:
        raise ValueError(
            f"{len(attack_names)} attack datasets do not fit in a batch of {batch_size}"
        )

    n_batches = len(labels_array) // batch_size
    if n_batches < 1:
        raise ValueError(
            f"Need at least {batch_size} rows to form one covered batch, got {len(labels_array)}"
        )

    target_attacks = batch_size // 2
    n_extra_attacks = max(0, target_attacks - len(attack_names))
    n_lives = batch_size - len(attack_names) - n_extra_attacks

    rng = np.random.default_rng(seed)
    groups: dict[str, np.ndarray] = {}
    for name in attack_names:
        idx = np.flatnonzero((labels_array == 1) & (dataset_array == name)).copy()
        rng.shuffle(idx)
        groups[name] = idx
    attack_pool = np.flatnonzero(labels_array == 1).copy()
    live_pool = np.flatnonzero(labels_array == 0).copy()
    rng.shuffle(attack_pool)
    rng.shuffle(live_pool)
    if n_lives > 0 and len(live_pool) == 0:
        raise ValueError("cover_attack_datasets needs label-0 rows to fill the live slots")

    group_pos = {name: 0 for name in attack_names}
    attack_pos = 0
    live_pos = 0
    batches = np.empty((n_batches, batch_size), dtype=np.int64)

    def take(pool: np.ndarray, pos: int, count: int, used: set[int]) -> tuple[list[int], int]:
        picked: list[int] = []
        if count <= 0 or len(pool) == 0:
            return picked, pos
        scanned = 0
        while len(picked) < count and scanned < len(pool):
            item = int(pool[pos % len(pool)])
            pos += 1
            scanned += 1
            if item in used:
                continue
            used.add(item)
            picked.append(item)
        return picked, pos

    for batch_index in range(n_batches):
        used: set[int] = set()
        chosen: list[int] = []
        for name in attack_names:
            idx = groups[name]
            if group_pos[name] > 0 and group_pos[name] % len(idx) == 0:
                rng.shuffle(idx)
            pick = int(idx[group_pos[name] % len(idx)])
            group_pos[name] += 1
            if pick in used:
                raise ValueError(f"attack dataset {name!r} repeated inside one batch")
            used.add(pick)
            chosen.append(pick)
        extra, attack_pos = take(attack_pool, attack_pos, n_extra_attacks, used)
        chosen.extend(extra)
        lives, live_pos = take(live_pool, live_pos, batch_size - len(chosen), used)
        chosen.extend(lives)
        if len(chosen) != batch_size:
            raise ValueError(
                f"Could not fill a batch of {batch_size} with one attack from each of "
                f"{len(attack_names)} datasets (got {len(chosen)} rows). "
                "A class is too small for this batch size."
            )
        rng.shuffle(chosen)
        batches[batch_index] = chosen
    return batches


def describe_attack_coverage(labels: list[int], datasets: list[str], batch_size: int) -> str:
    labels_array = np.asarray(labels, dtype=np.int32)
    dataset_array = np.asarray(datasets)
    names = sorted({str(name) for name, label in zip(dataset_array, labels_array) if int(label) == 1})
    n_batches = len(labels_array) // batch_size
    n_extra = max(0, batch_size // 2 - len(names))
    n_lives = batch_size - len(names) - n_extra
    lines = [
        f"Attack-dataset batches: {len(names)} datasets, 1 attack from each, "
        f"plus {n_extra} other attacks and {n_lives} lives; "
        f"{n_batches} steps of {batch_size}",
    ]
    for name in names:
        count = int(np.sum((labels_array == 1) & (dataset_array == name)))
        lines.append(
            f"  {name}: {count} attacks, about {n_batches / count:.1f} uses of each image per epoch"
        )
    return "\n".join(lines)


def make_attack_coverage_dataset(
    paths: list[str],
    labels: list[int],
    bboxes: list[str],
    datasets: list[str],
    batch_size: int,
    use_bbox_crop: bool,
    margin: float,
    bbox_aug_prob: float,
    seed: int,
) -> tf.data.Dataset:
    """Infinite dataset. Each epoch is a fresh plan; fit must set steps_per_epoch."""
    n_batches = len(paths) // batch_size
    epoch_length = n_batches * batch_size
    path_t = tf.constant(paths)
    label_t = tf.constant(labels, dtype=tf.int32)
    bbox_t = tf.constant(bboxes)

    def epochs():
        epoch_seed = seed
        while True:
            plan = plan_attack_coverage_batches(labels, datasets, batch_size, epoch_seed)
            epoch_seed += 1
            yield plan.reshape(-1).astype(np.int64)

    index_ds = tf.data.Dataset.from_generator(
        epochs,
        output_signature=tf.TensorSpec(shape=(epoch_length,), dtype=tf.int64),
    )
    dataset = index_ds.flat_map(tf.data.Dataset.from_tensor_slices)
    dataset = dataset.map(
        lambda index: (path_t[index], label_t[index], bbox_t[index]),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.map(
        lambda path, label, bbox: load_image(path, label, bbox, use_bbox_crop, margin, bbox_aug_prob, training=True),
        num_parallel_calls=AUTOTUNE,
    )
    return batch_training(dataset, batch_size, drop_remainder=True)


def make_dataset(
    paths: list[str],
    labels: list[int],
    bboxes: list[str],
    batch_size: int,
    training: bool,
    use_bbox_crop: bool,
    margin: float,
    bbox_aug_prob: float = 0.0,
    datasets: list[str] | None = None,
    cover_attack_datasets: bool = False,
    seed: int = 0,
) -> tf.data.Dataset:
    if training and cover_attack_datasets:
        if datasets is None:
            raise ValueError("cover_attack_datasets requires a dataset name for every training row")
        return make_attack_coverage_dataset(
            paths,
            labels,
            bboxes,
            datasets,
            batch_size,
            use_bbox_crop,
            margin,
            bbox_aug_prob,
            seed,
        )

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, bboxes))
    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    aug_prob = bbox_aug_prob if training else 0.0
    dataset = dataset.map(
        lambda path, label, bbox: load_image(path, label, bbox, use_bbox_crop, margin, aug_prob, training=training),
        num_parallel_calls=AUTOTUNE,
    )
    if training:
        return batch_training(dataset, batch_size, drop_remainder=False)
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


class TrainTimingCallback(tf.keras.callbacks.Callback):
    """Per-epoch wall time split: training steps, Keras validation pass, and the rest
    (EER callback predict over validation, checkpoint saves) -- so a slow run shows
    *where* it is slow. Must be last in the callbacks list so its on_epoch_end runs
    after the EER callback and the checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.train_seconds: list[float] = []
        self.val_seconds: list[float] = []
        self.epoch_seconds: list[float] = []
        self._t0 = 0.0
        self._t_val = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self._t0 = time.perf_counter()

    def on_test_begin(self, logs: dict | None = None) -> None:
        if self._t0 and len(self.train_seconds) == len(self.epoch_seconds):
            self.train_seconds.append(time.perf_counter() - self._t0)
        self._t_val = time.perf_counter()

    def on_test_end(self, logs: dict | None = None) -> None:
        if self._t_val and len(self.val_seconds) == len(self.epoch_seconds):
            self.val_seconds.append(time.perf_counter() - self._t_val)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if self._t0:
            self.epoch_seconds.append(time.perf_counter() - self._t0)
            self._t0 = 0.0


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


PATH_KEYS = ("csv", "validation_csv", "test_csv", "output_dir")
BOOL_KEYS = (
    "cosine_decay",
    "train_backbone",
    "require_gpu",
    "mixed_precision",
    "use_bbox_crop",
    "cover_attack_datasets",
)
CONFIG_DEFAULTS = {
    "csv": Path("test_df.csv"),
    "validation_csv": None,
    "test_csv": None,
    "output_dir": Path("runs/efficientnet_b2"),
    "epochs": 12,
    "batch_size": 32,
    "learning_rate": 1e-5,
    "cosine_decay": False,
    "min_learning_rate": 0.0,
    "validation_split": 0.1,
    "seed": 42,
    "train_backbone": False,
    "require_gpu": False,
    "mixed_precision": False,
    "eer_threshold": 0.5,
    "use_bbox_crop": False,
    "margin": 0.0,
    "bbox_aug_prob": 0.0,
    "cover_attack_datasets": False,
    "degrade": "",
    "augment": "base",
    "freq_aug": "",
    "checkpoint_monitor": "val_eer",
    "image_size": 512,
    "resize_mode": "squash",
    "comment": "",
}
CHECKPOINT_MODES = {"val_eer": "min", "val_auc": "max", "val_loss": "min", "val_acer": "min"}


def _normalize_config_key(key: str) -> str:
    return key.strip().replace("-", "_")


def load_config_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required for .yaml/.yml configs") from error
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON/YAML object, got {type(data).__name__}")

    unknown = []
    parsed = {}
    for raw_key, value in data.items():
        key = _normalize_config_key(str(raw_key))
        if key in {"config"}:
            continue
        if key not in CONFIG_DEFAULTS:
            unknown.append(raw_key)
            continue
        if key in PATH_KEYS:
            parsed[key] = None if value in (None, "") else Path(value)
        else:
            parsed[key] = value

    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}. Allowed: {sorted(CONFIG_DEFAULTS)}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB2 for binary liveness classification. Prefer --config; CLI flags override the file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON or YAML file with training settings. CLI flags override the file.",
    )
    parser.add_argument("--csv", type=Path, help="CSV with path,label columns.")
    parser.add_argument("--validation-csv", type=Path, help="Optional fixed validation CSV. If omitted, validation is split from --csv.")
    parser.add_argument("--test-csv", type=Path, help="Optional extra CSV used only for final testing and EER.")
    parser.add_argument("--output-dir", type=Path, help="Where to save model and reports.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cosine-decay", action=argparse.BooleanOptionalAction, default=None, help="Use cosine decay from --learning-rate to --min-learning-rate.")
    parser.add_argument("--min-learning-rate", type=float, help="Final LR floor for cosine decay.")
    parser.add_argument("--validation-split", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=None, help="Fine-tune EfficientNetB2 from the first epoch.")
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=None, help="Fail if TensorFlow cannot see a GPU.")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None, help="Use mixed precision, useful on NVIDIA T4.")
    parser.add_argument("--eer-threshold", type=float, help="Threshold for BPCER/APCER/ACER on sigmoid scores.")
    parser.add_argument("--use-bbox-crop", action=argparse.BooleanOptionalAction, default=None, help="Crop images by CSV bbox column before resizing.")
    parser.add_argument("--margin", type=float, help="BBox crop margin percent. 5 expands by 5%%, -5 crops 5%% inside.")
    parser.add_argument("--bbox-aug-prob", type=float, help="Probability of applying bbox crop as augmentation during training (0.0 to disable).")
    parser.add_argument(
        "--cover-attack-datasets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Every training batch includes one attack from each dataset that has label-1 rows.",
    )
    parser.add_argument(
        "--degrade",
        type=str,
        help="Comma-separated anonymisation degradations applied to every split before crop/resize: "
        "mask:BAND (fill document interior, keep BAND x bbox border), pixelate:N (document short side "
        "to N px and back), downscale:N (frame long side to N px and back), downscale_doc:N (frame scaled "
        "so the document long side is N px, and back). E.g. mask:0.05,downscale:192",
    )
    parser.add_argument(
        "--checkpoint-monitor",
        type=str,
        choices=sorted(CHECKPOINT_MODES),
        help="Validation metric that selects best.keras. Default val_eer: validation is production "
        "data and EER is the metric acted on; val_auc peaked at epoch 0 in most runs while EER did not.",
    )
    parser.add_argument("--augment", choices=sorted(AUGMENT_PRESETS), help="Training augmentation preset: base (flip/affine/noise) or domain (adds photometric, blur, noise, JPEG, perspective, shadow).")
    parser.add_argument(
        "--freq-aug",
        type=str,
        help="Comma-separated training-only frequency augmentations against capture-pipeline shortcuts: "
        "rescale:P (random intermediate resolution + JPEG before the network resize), bandstop:P (attenuate "
        "a random spectral ring), ampmix:P[:ETA] (mix Fourier amplitude with an image of the other class, "
        "keep phase). E.g. rescale:0.5,bandstop:0.3,ampmix:0.5",
    )
    parser.add_argument("--image-size", type=int, help="Network input side in px (default 512). Exported doc:96 frames are ~170-210 px.")
    parser.add_argument(
        "--resize-mode",
        type=str,
        choices=RESIZE_MODES,
        help="squash: bilinear to the square ignoring aspect (default). letterbox: downscale only if larger, "
        "never upscale, pad with the frame mean -- frames that fit are passed through pixel for pixel.",
    )
    parser.add_argument("--comment", type=str, help="Free-text note saved to report.json.")
    args = parser.parse_args()

    merged = dict(CONFIG_DEFAULTS)
    if args.config is not None:
        merged.update(load_config_file(args.config))

    for key in CONFIG_DEFAULTS:
        cli_value = getattr(args, key)
        if cli_value is not None:
            merged[key] = cli_value

    for key in BOOL_KEYS:
        merged[key] = bool(merged[key])

    resolved = argparse.Namespace(**merged, config=args.config)
    print("Resolved training config:")
    print(json.dumps(config_as_dict(resolved), indent=2))
    return resolved


def config_as_dict(args: argparse.Namespace) -> dict:
    payload = {}
    for key in CONFIG_DEFAULTS:
        value = getattr(args, key)
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    if args.config is not None:
        payload["config"] = str(args.config)
    return payload


def build_callbacks(args: argparse.Namespace, validation_paths: list[str], validation_labels: list[int],
                    validation_bboxes: list[str]) -> list[tf.keras.callbacks.Callback]:
    """EERCallback must run before the checkpoints: it writes val_eer into `logs`,
    and ModelCheckpoint reads the monitor from the same dict in the same epoch."""
    monitor = args.checkpoint_monitor
    mode = CHECKPOINT_MODES[monitor]
    return [
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best.keras"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "checkpoints" / f"epoch_{{epoch:03d}}_{monitor}_{{{monitor}:.4f}}.keras"),
            monitor=monitor,
            mode=mode,
            save_best_only=False,
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


def best_epoch(history: dict, monitor: str) -> int | None:
    values = history.get(monitor)
    if not values:
        return None
    clean = [(v, i) for i, v in enumerate(values) if v == v]  # drop NaN
    if not clean:
        return None
    return (min if CHECKPOINT_MODES[monitor] == "min" else max)(clean)[1]


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    configure_runtime(args.require_gpu, args.mixed_precision)
    try:
        set_input_geometry(args.image_size, args.resize_mode)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Input geometry: {IMAGE_SIZE}x{IMAGE_SIZE}, {RESIZE_MODE}")
    set_augment_preset(args.augment)
    print(f"Augmentation preset: {args.augment}")

    try:
        DEGRADATIONS[:] = parse_degrade(args.degrade or "")
    except ValueError as error:
        raise SystemExit(f"--degrade: {error}") from error
    if DEGRADATIONS:
        print("Degradations:", ", ".join(f"{kind}:{value:g}" for kind, value in DEGRADATIONS))
    try:
        FREQ_AUG.clear()
        FREQ_AUG.update(parse_freq_aug(args.freq_aug or ""))
    except ValueError as error:
        raise SystemExit(f"--freq-aug: {error}") from error
    if FREQ_AUG:
        print("Frequency augmentations:", ", ".join(f"{kind}:{prob:g}" + (f":{param:g}" if kind == "ampmix" else "")
                                                    for kind, (prob, param) in FREQ_AUG.items()))
    degrade_needs_bbox = any(kind in ("mask", "pixelate", "downscale_doc") for kind, _ in DEGRADATIONS)
    require_bbox = args.use_bbox_crop or degrade_needs_bbox

    def check_degrade_bboxes(csv_path: Path, bbox_values: list[str]) -> None:
        # A row without a bbox would pass through mask/pixelate untouched, i.e.
        # un-anonymised -- and if that happens to one class more than the other
        # the model learns it. Refuse rather than warn.
        if not degrade_needs_bbox:
            return
        empty = sum(1 for value in bbox_values if not value)
        if empty:
            raise SystemExit(
                f"{csv_path}: {empty} rows have no bbox but --degrade "
                f"{args.degrade!r} needs one for every row"
            )

    loaded = read_csv_dataset(
        args.csv,
        require_bbox=require_bbox,
        with_dataset=args.cover_attack_datasets,
    )
    if args.cover_attack_datasets:
        paths, labels, bboxes, datasets = loaded
    else:
        paths, labels, bboxes = loaded
        datasets = None
    check_degrade_bboxes(args.csv, bboxes)
    if args.validation_csv:
        train_paths = paths
        train_labels = labels
        train_bboxes = bboxes
        train_datasets = datasets
        validation_paths, validation_labels, validation_bboxes = read_csv_dataset(
            args.validation_csv,
            require_bbox=require_bbox,
        )
        check_degrade_bboxes(args.validation_csv, validation_bboxes)
        validation_csv = args.validation_csv
    else:
        train_indices, validation_indices = split_indices(labels, args.validation_split, args.seed)
        train_paths = [paths[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        train_bboxes = [bboxes[index] for index in train_indices]
        train_datasets = [datasets[index] for index in train_indices] if datasets is not None else None
        validation_paths = [paths[index] for index in validation_indices]
        validation_labels = [labels[index] for index in validation_indices]
        validation_bboxes = [bboxes[index] for index in validation_indices]
        validation_csv = None

    if args.cover_attack_datasets:
        print(describe_attack_coverage(train_labels, train_datasets, args.batch_size))
    train_dataset = make_dataset(
        train_paths,
        train_labels,
        train_bboxes,
        args.batch_size,
        training=True,
        use_bbox_crop=args.use_bbox_crop,
        margin=args.margin,
        bbox_aug_prob=args.bbox_aug_prob,
        datasets=train_datasets,
        cover_attack_datasets=args.cover_attack_datasets,
        seed=args.seed,
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
            require_bbox=require_bbox,
        )
        check_degrade_bboxes(args.test_csv, test_bboxes)
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

    if args.cover_attack_datasets:
        train_steps_per_epoch = len(train_paths) // args.batch_size
    else:
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
    if args.config is not None:
        shutil.copy2(args.config, args.output_dir / args.config.name)
    with (args.output_dir / "train_config.json").open("w", encoding="utf-8") as file:
        json.dump(config_as_dict(args), file, indent=2)

    callbacks = build_callbacks(args, validation_paths, validation_labels, validation_bboxes)
    timing = TrainTimingCallback()
    callbacks.append(timing)

    print(f"Starting training for requested epochs: {args.epochs}")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        steps_per_epoch=train_steps_per_epoch if args.cover_attack_datasets else None,
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
        "degrade": [{"kind": kind, "value": value} for kind, value in DEGRADATIONS],
        "freq_aug": [
            {"kind": kind, "prob": prob, **({"eta": param} if kind == "ampmix" else {})}
            for kind, (prob, param) in FREQ_AUG.items()
        ],
        "input": {"image_size": IMAGE_SIZE, "resize_mode": RESIZE_MODE},
        "throughput": {
            "train_seconds_per_epoch": [round(v, 1) for v in timing.train_seconds],
            "keras_validation_seconds_per_epoch": [round(v, 1) for v in timing.val_seconds],
            "other_seconds_per_epoch": [
                round(e - t - v, 1) for e, t, v in zip(timing.epoch_seconds, timing.train_seconds, timing.val_seconds)
            ],  # EER callback predict over validation + checkpoint saves
            "epoch_wall_seconds": [round(v, 1) for v in timing.epoch_seconds],
            "train_images_per_sec": round(len(train_paths) / (sum(timing.train_seconds) / len(timing.train_seconds)), 1)
            if timing.train_seconds else None,
            "steady_state_train_images_per_sec": round(len(train_paths) / min(timing.train_seconds), 1)
            if timing.train_seconds else None,
            "fit_wall_seconds": round(sum(timing.epoch_seconds), 1),
        },
        "checkpoint": {
            "monitor": args.checkpoint_monitor,
            "mode": CHECKPOINT_MODES[args.checkpoint_monitor],
            "best_epoch": best_epoch(history.history, args.checkpoint_monitor),
        },
        "comment": args.comment,
        "config": config_as_dict(args),
        "test_metrics": metrics,
        "test_eer_metrics": eer_metrics,
        "history": history.history,
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "test_metrics": metrics,
        "test_eer_metrics": eer_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
