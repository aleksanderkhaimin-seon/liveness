#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_model_config(model_dir: Path) -> dict:
    config_path = model_dir / f"{model_dir.name}.json"
    if not config_path.exists():
        candidates = sorted(model_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No JSON model config found in {model_dir}")
        config_path = candidates[0]

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_model_path(model_dir: Path, config: dict) -> Path:
    model_name = config.get("model_name")
    if model_name:
        model_path = model_dir / model_name
        if model_path.exists():
            return model_path

    candidates = sorted(model_dir.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No ONNX model found in {model_dir}")
    return candidates[0]


def input_layout(input_shape: list) -> str:
    shape = [dim if isinstance(dim, int) else None for dim in input_shape]

    if len(shape) != 4:
        return "NHWC"
    if shape[1] in (1, 3, 4):
        return "NCHW"
    if shape[3] in (1, 3, 4):
        return "NHWC"

    return "NHWC"


def interpolation_mode(mode: str) -> int:
    if mode == "INTER_LINEAR":
        return Image.Resampling.BILINEAR
    if mode == "INTER_CUBIC":
        return Image.Resampling.BICUBIC
    return Image.Resampling.NEAREST


def normalize(array: np.ndarray, step: dict) -> np.ndarray:
    mean = np.asarray(step["mean"], dtype=np.float32)
    std = np.asarray(step["std"], dtype=np.float32)

    if mean.size != array.shape[-1] or std.size != array.shape[-1]:
        raise ValueError(
            f"Normalize mean/std must have {array.shape[-1]} values, "
            f"got mean={mean.size}, std={std.size}"
        )

    return (array - mean) / std


def apply_preprocessors(image: Image.Image, config: dict) -> np.ndarray:
    current_image = image.convert("RGB")
    array = None

    for step in config.get("preprocessors", []):
        step_type = step.get("type")

        if step_type == "resize":
            width = int(step["target_width"])
            height = int(step["target_height"])
            interpolation = interpolation_mode(step.get("interpolation_mode", "INTER_NEAREST"))
            current_image = current_image.resize((width, height), interpolation)
            array = None
            continue

        if array is None:
            array = np.asarray(current_image, dtype=np.float32)

        if step_type == "convert_to_float":
            array = array.astype(np.float32, copy=False)
        elif step_type == "normalize":
            array = normalize(array.astype(np.float32, copy=False), step)
        elif step_type == "lead_to_orientation":
            continue
        else:
            raise ValueError(f"Unsupported preprocessor type: {step_type}")

    if array is None:
        array = np.asarray(current_image, dtype=np.float32)

    return array.astype(np.float32, copy=False)


def preprocess_image(image_path: Path, config: dict, layout: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    array = apply_preprocessors(image, config)

    if layout == "NCHW":
        array = np.transpose(array, (2, 0, 1))

    return np.expand_dims(array, axis=0)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def summarize_output(output: np.ndarray, labels: list[str] | None, output_activation: str) -> dict:
    values = np.asarray(output).squeeze()
    flat = values.reshape(-1).astype(float)

    result = {
        "raw": flat.tolist(),
    }

    if flat.size == 1:
        raw_score = flat[0]
        if output_activation == "linear":
            probability = sigmoid(raw_score)
        elif output_activation == "sigmoid":
            probability = raw_score
        else:
            probability = raw_score if 0.0 <= raw_score <= 1.0 else sigmoid(raw_score)

        result["score"] = probability
        result["prediction"] = labels[1] if labels and len(labels) > 1 and probability >= 0.5 else None
        return result

    probabilities = flat
    if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)) or not np.isclose(np.sum(probabilities), 1.0, atol=1e-3):
        probabilities = softmax(flat)

    predicted_index = int(np.argmax(probabilities))
    result["probabilities"] = probabilities.tolist()
    result["predicted_index"] = predicted_index
    if labels and predicted_index < len(labels):
        result["prediction"] = labels[predicted_index]

    return result


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = [path for path in sorted(input_path.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with an ONNX model folder.")
    parser.add_argument("input", type=Path, help="Image file or directory with images.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/m1"), help="Folder containing the .onnx and .json files.")
    parser.add_argument("--labels", nargs="*", help="Optional class labels in model output order.")
    parser.add_argument("--providers", nargs="*", default=["CPUExecutionProvider"], help="ONNX Runtime providers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.model_dir)
    model_path = resolve_model_path(args.model_dir, config)
    output_activation = config.get("output_activation", "auto")

    session = ort.InferenceSession(str(model_path), providers=args.providers)
    session_inputs = {model_input.name: model_input for model_input in session.get_inputs()}
    session_outputs = {model_output.name: model_output for model_output in session.get_outputs()}

    input_name = config.get("nnet_input_name")
    if input_name not in session_inputs:
        input_name = session.get_inputs()[0].name

    output_name = config.get("nnet_output_name")
    if output_name not in session_outputs:
        output_name = session.get_outputs()[0].name

    model_input = session_inputs[input_name]
    layout = input_layout(model_input.shape)

    for image_path in iter_images(args.input):
        tensor = preprocess_image(image_path, config, layout)
        output = session.run([output_name], {input_name: tensor})[0]
        result = summarize_output(output, args.labels, output_activation)
        result["image"] = str(image_path)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
