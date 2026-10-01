#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx

import train_efficientnet_b2 as teb
from input_geometry import find_report, preprocessor_steps, resolve_geometry
from train_efficientnet_b2 import build_model


DEFAULT_INPUT_NAME = "x"
DEFAULT_OUTPUT_NAME = "Identity"
IMAGE_SIZE = 512  # default only; the checkpoint's input shape is authoritative


def iter_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from iter_layers(layer)


def build_inference_model(image_size: int = IMAGE_SIZE):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name=DEFAULT_INPUT_NAME)
    backbone = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=None,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    x = backbone(inputs, training=False)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32", name="live_score")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def copy_matching_weights(source_model, target_model) -> None:
    source_layers = {layer.name: layer for layer in iter_layers(source_model)}
    copied = []
    skipped = []

    for target_layer in iter_layers(target_model):
        target_weights = target_layer.get_weights()
        if not target_weights:
            continue

        source_layer = source_layers.get(target_layer.name)
        if source_layer is None:
            skipped.append(target_layer.name)
            continue

        source_weights = source_layer.get_weights()
        if len(source_weights) != len(target_weights):
            skipped.append(target_layer.name)
            continue

        if any(src.shape != dst.shape for src, dst in zip(source_weights, target_weights)):
            skipped.append(target_layer.name)
            continue

        target_layer.set_weights(source_weights)
        copied.append(target_layer.name)

    if skipped:
        print(f"Skipped {len(skipped)} weighted layers with no compatible source weights.", file=sys.stderr)
    print(f"Copied weights for {len(copied)} weighted layers into inference model.", file=sys.stderr)


def build_legacy_training_model(image_size: int = IMAGE_SIZE):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.03)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)

    backbone = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling="avg",
    )
    backbone.trainable = True

    x = backbone.output
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32", name="live_score")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def extract_keras_weights(checkpoint_path: Path, work_dir: Path) -> Path | None:
    if checkpoint_path.suffix != ".keras" or not zipfile.is_zipfile(checkpoint_path):
        return None

    with zipfile.ZipFile(checkpoint_path) as archive:
        names = archive.namelist()
        if "model.weights.h5" not in names:
            return None

        extracted_path = work_dir / "model.weights.h5"
        with archive.open("model.weights.h5") as source, extracted_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    return extracted_path


def try_load_weights(model, weights_path: Path) -> bool:
    try:
        model.load_weights(weights_path)
        return True
    except Exception as error:
        print(f"Could not load weights from {weights_path}: {error}", file=sys.stderr)
        return False


def load_checkpoint_model(checkpoint_path: Path, image_size_hint: int = IMAGE_SIZE):
    """Load the .keras file; fall back to rebuilding the architecture at image_size_hint and loading weights."""
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(checkpoint_path, compile=False)
    except Exception as error:
        print(
            "Full-model load failed, trying architecture reconstruction plus weight load.",
            file=sys.stderr,
        )
        print(error, file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="liveness_weights_") as temp_dir:
        candidates = [checkpoint_path]
        extracted_weights = extract_keras_weights(checkpoint_path, Path(temp_dir))
        if extracted_weights:
            candidates.append(extracted_weights)

        teb.set_input_geometry(image_size_hint, "squash")  # build_model reads the module geometry
        builders = [
            lambda: build_model(learning_rate=1e-4, train_backbone=True, backbone_weights=None),
            lambda: build_legacy_training_model(image_size_hint),
        ]

        for weights_path in candidates:
            for builder in builders:
                model = builder()
                if try_load_weights(model, weights_path):
                    return model

    raise RuntimeError(f"Could not load model or weights from {checkpoint_path}")


def export_saved_model(model, export_dir: Path, image_size: int) -> None:
    import tensorflow as tf

    input_signature = [
        tf.TensorSpec(
            shape=(None, image_size, image_size, 3),
            dtype=tf.float32,
            name=DEFAULT_INPUT_NAME,
        )
    ]

    export_archive = tf.keras.export.ExportArchive()
    export_archive.track(model)
    export_archive.add_endpoint("serving_default", model.call, input_signature=input_signature)
    export_archive.write_out(str(export_dir))


def run_tf2onnx(saved_model_dir: Path, onnx_path: Path, opset: int) -> None:
    command = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        str(saved_model_dir),
        "--signature_def",
        "serving_default",
        "--opset",
        str(opset),
        "--output",
        str(onnx_path),
    ]
    subprocess.run(command, check=True)


def graph_io_names(onnx_path: Path) -> tuple[str, str]:
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    return input_name, output_name


def write_model_config(output_dir: Path, model_name: str, input_name: str, output_name: str, geometry) -> Path:
    config = {
        "model_name": model_name,
        "nnet_input_name": input_name,
        "nnet_output_name": output_name,
        "output_activation": "linear",
        "input": {"image_size": geometry.image_size, "resize_mode": geometry.resize_mode},
        # resize (squash) or letterbox step, then convert_to_float -- infer_onnx.py and
        # predict_onnx_csv.py read the mode back from here.
        "preprocessors": preprocessor_steps(geometry.image_size, geometry.resize_mode),
    }

    config_path = output_dir / f"{output_dir.name}.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")

    return config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Keras checkpoint to ONNX plus model JSON metadata.")
    parser.add_argument("checkpoint", type=Path, help="Path to a .keras checkpoint, for example runs/efficientnet_b2/best.keras.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output model folder, for example models/m4.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it already exists.")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Only needed when the .keras file cannot be loaded whole and the architecture is rebuilt; otherwise taken from the checkpoint.")
    parser.add_argument("--resize-mode", choices=("squash", "letterbox"), default=None,
                        help="Preprocessing written into the model JSON. Default: from report.json next to the checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint
    output_dir = args.output_dir

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    onnx_path = output_dir / f"{output_dir.name}.onnx"

    report_path = find_report(checkpoint_path)
    provisional = resolve_geometry(args.image_size, args.resize_mode, report_path=report_path)
    model = load_checkpoint_model(checkpoint_path, provisional.image_size)
    try:
        geometry = resolve_geometry(
            args.image_size, args.resize_mode, model_input_size=int(model.input_shape[1]), report_path=report_path,
        )
    except ValueError as error:
        raise SystemExit(f"Input geometry: {error}") from error
    print(f"Input geometry: {geometry}")

    with tempfile.TemporaryDirectory(prefix="liveness_saved_model_") as temp_dir:
        saved_model_dir = Path(temp_dir) / "saved_model"
        export_saved_model(model, saved_model_dir, geometry.image_size)
        run_tf2onnx(saved_model_dir, onnx_path, args.opset)

    input_name, output_name = graph_io_names(onnx_path)
    config_path = write_model_config(output_dir, onnx_path.name, input_name, output_name, geometry)

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "config": str(config_path),
        "image_size": geometry.image_size,
        "resize_mode": geometry.resize_mode,
        "input_name": input_name,
        "output_name": output_name,
    }, indent=2))


if __name__ == "__main__":
    main()
