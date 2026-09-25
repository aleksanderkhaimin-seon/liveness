#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ["background", "document", "photo"]
BACKGROUND_MODEL_CLASS_IDS = {0}
DISPLAY_CLASS_ID_OFFSET = 0
INPUT_NAME = "input:0"
BOXES_OUTPUT_NAME = "Identity_2:0"
SCORES_OUTPUT_NAME = "Identity_1:0"


def load_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: onnxruntime\n"
            "Install it with:\n"
            "  python3 -m pip install onnxruntime"
        ) from exc
    return ort


def preprocess_image(image_path, width=640, height=640):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    array = (array - 127.5) / 127.5
    return image, array[np.newaxis, ...], original_size


def decode_boxes(raw_boxes, anchors):
    """Decode TF Object Detection SSD box encodings.

    Both raw_boxes and anchors are expected in [y_center, x_center, h, w] order.
    The returned boxes are [ymin, xmin, ymax, xmax], normalized to the image.
    """
    y_center = raw_boxes[:, 0] / 10.0 * anchors[:, 2] + anchors[:, 0]
    x_center = raw_boxes[:, 1] / 10.0 * anchors[:, 3] + anchors[:, 1]
    height = np.exp(raw_boxes[:, 2] / 5.0) * anchors[:, 2]
    width = np.exp(raw_boxes[:, 3] / 5.0) * anchors[:, 3]

    ymin = y_center - height / 2.0
    xmin = x_center - width / 2.0
    ymax = y_center + height / 2.0
    xmax = x_center + width / 2.0
    boxes = np.stack([ymin, xmin, ymax, xmax], axis=1)
    return np.clip(boxes, 0.0, 1.0)


def box_iou(box, boxes):
    ymin = np.maximum(box[0], boxes[:, 0])
    xmin = np.maximum(box[1], boxes[:, 1])
    ymax = np.minimum(box[2], boxes[:, 2])
    xmax = np.minimum(box[3], boxes[:, 3])

    inter_h = np.maximum(0.0, ymax - ymin)
    inter_w = np.maximum(0.0, xmax - xmin)
    intersection = inter_h * inter_w

    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms(boxes, scores, iou_threshold):
    order = scores.argsort()[::-1]
    keep = []

    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break

        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]

    return keep


def normalized_to_pixels(box, image_width, image_height):
    ymin, xmin, ymax, xmax = box
    return [
        int(round(xmin * image_width)),
        int(round(ymin * image_height)),
        int(round(xmax * image_width)),
        int(round(ymax * image_height)),
    ]


def print_score_debug(scores, threshold):
    print("Score debug:", file=sys.stderr)
    for model_class_id, class_name in enumerate(CLASS_NAMES):
        class_scores = scores[:, model_class_id]
        display_class_id = model_class_id + DISPLAY_CLASS_ID_OFFSET
        print(
            "  "
            f"display class {display_class_id} / model column {model_class_id} "
            f"({class_name}): max={class_scores.max():.6f}, "
            f"count>={threshold:g}={np.count_nonzero(class_scores >= threshold)}",
            file=sys.stderr,
        )

    foreground_class_ids = [
        class_id for class_id in range(len(CLASS_NAMES))
        if class_id not in BACKGROUND_MODEL_CLASS_IDS
    ]
    foreground_scores = scores[:, foreground_class_ids]
    best_foreground_offsets = np.argmax(foreground_scores, axis=1)
    best_model_class_ids = np.asarray(foreground_class_ids)[best_foreground_offsets]
    best_scores = scores[np.arange(scores.shape[0]), best_model_class_ids]

    for model_class_id in foreground_class_ids:
        class_name = CLASS_NAMES[model_class_id]
        count = np.count_nonzero(
            (best_model_class_ids == model_class_id) & (best_scores >= threshold)
        )
        display_class_id = model_class_id + DISPLAY_CLASS_ID_OFFSET
        print(
            f"  best-class candidates for display class {display_class_id} "
            f"({class_name}): {count}",
            file=sys.stderr,
        )


def run_detector(
    model_path,
    anchors_path,
    image_path,
    threshold,
    iou_threshold,
    max_detections,
    debug_scores=False,
):
    ort = load_onnxruntime()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    image, input_tensor, (image_width, image_height) = preprocess_image(image_path)
    anchors = np.fromfile(anchors_path, dtype=np.float32).reshape(-1, 4)

    raw_boxes, scores = session.run(
        [BOXES_OUTPUT_NAME, SCORES_OUTPUT_NAME],
        {INPUT_NAME: input_tensor},
    )

    raw_boxes = np.squeeze(raw_boxes, axis=0)
    scores = np.squeeze(scores, axis=0)

    if raw_boxes.shape[0] != anchors.shape[0]:
        raise RuntimeError(
            f"Model returned {raw_boxes.shape[0]} boxes, but {anchors_path} has "
            f"{anchors.shape[0]} anchors."
        )

    if scores.shape[1] != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model returned {scores.shape[1]} score columns, but the script knows "
            f"{len(CLASS_NAMES)} classes."
        )

    if debug_scores:
        print_score_debug(scores, threshold)

    foreground_class_ids = [
        class_id for class_id in range(len(CLASS_NAMES))
        if class_id not in BACKGROUND_MODEL_CLASS_IDS
    ]
    if not foreground_class_ids:
        raise RuntimeError("All configured classes are marked as background.")

    boxes = decode_boxes(raw_boxes, anchors)
    foreground_scores = scores[:, foreground_class_ids]
    best_foreground_offsets = np.argmax(foreground_scores, axis=1)
    best_model_class_ids = np.asarray(foreground_class_ids)[best_foreground_offsets]
    best_scores = scores[np.arange(scores.shape[0]), best_model_class_ids]
    candidate_indices = np.flatnonzero(best_scores >= threshold)

    if candidate_indices.size == 0:
        return image, []

    selected = nms(boxes[candidate_indices], best_scores[candidate_indices], iou_threshold)
    detections = []

    for selected_idx in selected:
        anchor_idx = candidate_indices[selected_idx]
        model_class_id = int(best_model_class_ids[anchor_idx])
        display_class_id = model_class_id + DISPLAY_CLASS_ID_OFFSET
        normalized_box = boxes[anchor_idx]
        detections.append(
            {
                "label": CLASS_NAMES[model_class_id],
                "class_id": display_class_id,
                "model_class_id": model_class_id,
                "score": float(best_scores[anchor_idx]),
                "box": normalized_to_pixels(
                    normalized_box, image_width=image_width, image_height=image_height
                ),
                "normalized_box": [float(v) for v in normalized_box],
            }
        )

    detections.sort(key=lambda item: item["score"], reverse=True)
    return image, detections[:max_detections]


def draw_detections(image, detections, output_path):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f'class {detection["class_id"]} {detection["label"]} {detection["score"]:.2f}'
        color = "lime" if detection["label"] == "document" else "deepskyblue"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        draw.rectangle([x1, max(0, y1 - text_h - 4), x1 + text_w + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - text_h - 2)), label, fill="black", font=font)

    image.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run detector_01 ONNX inference.")
    parser.add_argument("image", type=Path, help="Path to the image to inspect.")
    parser.add_argument("--model", type=Path, default=Path("detector_01.onnx"))
    parser.add_argument("--anchors", type=Path, default=Path("anchors.bin"))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument(
        "--debug-scores",
        action="store_true",
        help="Print max score and candidate counts for each class.",
    )
    parser.add_argument("--output", type=Path, help="Optional annotated image output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    image, detections = run_detector(
        model_path=args.model,
        anchors_path=args.anchors,
        image_path=args.image,
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
        debug_scores=args.debug_scores,
    )

    print(json.dumps(detections, indent=2))

    if args.output:
        draw_detections(image, detections, args.output)
        print(f"\nSaved annotated image to: {args.output}")


if __name__ == "__main__":
    main()
