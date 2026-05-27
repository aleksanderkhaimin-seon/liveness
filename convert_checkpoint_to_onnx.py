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

from train_efficientnet_b2 import build_model


DEFAULT_INPUT_NAME = "x"
DEFAULT_OUTPUT_NAME = "Identity"
IMAGE_SIZE = 512


def iter_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from iter_layers(layer)


def build_inference_model():
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name=DEFAULT_INPUT_NAME)
    backbone = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=None,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
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


def build_legacy_training_model():
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
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


def load_checkpoint_model(checkpoint_path: Path):
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

        builders = [
            lambda: build_model(learning_rate=1e-4, train_backbone=True, backbone_weights=None),
            build_legacy_training_model,
        ]

        for weights_path in candidates:
            for builder in builders:
                model = builder()
                if try_load_weights(model, weights_path):
                    return model

    raise RuntimeError(f"Could not load model or weights from {checkpoint_path}")


def export_saved_model(checkpoint_path: Path, export_dir: Path) -> None:
    import tensorflow as tf

    checkpoint_model = load_checkpoint_model(checkpoint_path)
    model = build_inference_model()
    copy_matching_weights(checkpoint_model, model)

    input_signature = [
        tf.TensorSpec(
            shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3),
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


def write_model_config(output_dir: Path, model_name: str, input_name: str, output_name: str) -> Path:
    config = {
        "model_name": model_name,
        "nnet_input_name": input_name,
        "nnet_output_name": output_name,
        "output_activation": "linear",
        "preprocessors": [
            {
                "type": "resize",
                "target_width": IMAGE_SIZE,
                "target_height": IMAGE_SIZE,
                "interpolation_mode": "INTER_LINEAR",
            },
            {
                "type": "convert_to_float",
            },
        ],
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

    with tempfile.TemporaryDirectory(prefix="liveness_saved_model_") as temp_dir:
        saved_model_dir = Path(temp_dir) / "saved_model"
        export_saved_model(checkpoint_path, saved_model_dir)
        run_tf2onnx(saved_model_dir, onnx_path, args.opset)

    input_name, output_name = graph_io_names(onnx_path)
    config_path = write_model_config(output_dir, onnx_path.name, input_name, output_name)

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "config": str(config_path),
        "input_name": input_name,
        "output_name": output_name,
    }, indent=2))


if __name__ == "__main__":
    main()
