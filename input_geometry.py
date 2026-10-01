#!/usr/bin/env python3
"""Input geometry shared by the inference-side scripts.

Training (train_efficientnet_b2.py) decides two things about the network
input: the square side `image_size` and how a frame is brought to it,
`resize_mode` (squash | letterbox). Everything downstream must apply the
same choice, so this module centralises:

  * where to find it -- the model's own input shape for the size, and the
    run's report.json (next to the checkpoint) or the exported model's JSON
    config for the mode;
  * how to apply it in PIL/numpy, mirroring fit_to_input() in the training
    script: squash = bilinear resize ignoring aspect; letterbox = area
    downscale only if the frame is larger, never upscale, pad centred with
    the frame's per-channel mean, frame copied exactly otherwise.

No TensorFlow dependency, so the ONNX tools can import it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_IMAGE_SIZE = 512
DEFAULT_RESIZE_MODE = "squash"
RESIZE_MODES = ("squash", "letterbox")


@dataclass(frozen=True)
class InputGeometry:
    image_size: int
    resize_mode: str
    source: str  # where the values came from, for the log line

    def __str__(self) -> str:
        return f"{self.image_size}x{self.image_size}, {self.resize_mode} ({self.source})"


# -- discovery ----------------------------------------------------------------

def find_report(checkpoint: Path) -> Path | None:
    """report.json next to a checkpoint, or one level up for checkpoints/epoch_*.keras."""
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        candidate = directory / "report.json"
        if candidate.is_file():
            return candidate
    return None


def geometry_from_report(report_path: Path) -> tuple[int | None, str | None]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    block = data.get("input") or {}
    config = data.get("config") or {}
    size = block.get("image_size", config.get("image_size"))
    mode = block.get("resize_mode", config.get("resize_mode"))
    return (int(size) if size else None, str(mode) if mode else None)


def geometry_from_model_config(config: dict) -> tuple[int | None, str | None]:
    """From an exported model's JSON (write_model_config in convert_checkpoint_to_onnx.py)."""
    block = config.get("input") or {}
    size = block.get("image_size")
    mode = block.get("resize_mode")
    if size is None or mode is None:
        for step in config.get("preprocessors", []):
            if step.get("type") in ("resize", "letterbox"):
                size = size or step.get("target_width")
                mode = mode or ("letterbox" if step["type"] == "letterbox" else "squash")
    return (int(size) if size else None, mode)


def resolve_geometry(
    image_size: int | None = None,
    resize_mode: str | None = None,
    *,
    model_input_size: int | None = None,
    report_path: Path | None = None,
    model_config: dict | None = None,
) -> InputGeometry:
    """Explicit flags win; then the model's own shape (size) and report/config (mode); then defaults.

    A flag that contradicts the model's input shape is an error: the network
    cannot take another size.
    """
    report_size = report_mode = None
    if report_path is not None and report_path.is_file():
        report_size, report_mode = geometry_from_report(report_path)
    config_size = config_mode = None
    if model_config:
        config_size, config_mode = geometry_from_model_config(model_config)

    size_source = "flag"
    size = image_size
    if size is None:
        for value, name in ((model_input_size, "model input shape"), (config_size, "model config"), (report_size, "report.json")):
            if value:
                size, size_source = int(value), name
                break
    if size is None:
        size, size_source = DEFAULT_IMAGE_SIZE, "default"
    if model_input_size and size != model_input_size:
        raise ValueError(f"image_size {size} ({size_source}) does not match the model input {model_input_size}x{model_input_size}")

    mode_source = "flag"
    mode = resize_mode
    if mode is None:
        for value, name in ((config_mode, "model config"), (report_mode, "report.json")):
            if value:
                mode, mode_source = value, name
                break
    if mode is None:
        mode, mode_source = DEFAULT_RESIZE_MODE, "default"
        if size != DEFAULT_IMAGE_SIZE:
            print(
                f"WARNING: image_size {size} but no resize_mode recorded; assuming {mode}. "
                "Pass --resize-mode if the model was trained with letterbox.",
                file=sys.stderr,
            )
    if mode not in RESIZE_MODES:
        raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {mode!r}")

    source = size_source if size_source == mode_source else f"size: {size_source}, mode: {mode_source}"
    return InputGeometry(image_size=size, resize_mode=mode, source=source)


# -- application --------------------------------------------------------------

def fit_to_input_pil(image: Image.Image, image_size: int, resize_mode: str) -> Image.Image:
    """PIL mirror of train_efficientnet_b2.fit_to_input()."""
    image = image.convert("RGB")
    if resize_mode == "squash":
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if resize_mode != "letterbox":
        raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {resize_mode!r}")

    scale = min(1.0, image_size / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BOX,
        )
    width, height = image.size
    mean = tuple(int(round(v)) for v in np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0))
    canvas = Image.new("RGB", (image_size, image_size), mean)
    canvas.paste(image, ((image_size - width) // 2, (image_size - height) // 2))
    return canvas


def fit_to_input_array(image: Image.Image, image_size: int, resize_mode: str) -> np.ndarray:
    """float32 [H, W, 3] in [0, 255], as the network expects."""
    return np.asarray(fit_to_input_pil(image, image_size, resize_mode), dtype=np.float32)


def preprocessor_steps(image_size: int, resize_mode: str) -> list[dict]:
    """The preprocessing chain for an exported model's JSON config."""
    if resize_mode == "letterbox":
        geometry_step = {
            "type": "letterbox",
            "target_width": image_size,
            "target_height": image_size,
            "downscale_interpolation": "INTER_AREA",
            "upscale": False,
            "fill": "mean",
        }
    else:
        geometry_step = {
            "type": "resize",
            "target_width": image_size,
            "target_height": image_size,
            "interpolation_mode": "INTER_LINEAR",
        }
    return [geometry_step, {"type": "convert_to_float"}]
