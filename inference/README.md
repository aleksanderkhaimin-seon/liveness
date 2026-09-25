# Detection Models

This folder contains ONNX detector models and helper files for document/image detection.

## Files

- `detector_01.onnx` - main detector model.
- `anchors.bin` - SSD anchors used to decode model boxes.
- `detector_01.json` - model configuration and preprocessing details.
- `run_detector_01.py` - Python inference script for `detector_01.onnx`.

## Class IDs

`detector_01.onnx` returns 3 score columns:

- `0` - background / empty frame
- `1` - document
- `2` - photo

The inference script ignores class `0` and returns only foreground detections.

## Run Inference

Install dependencies:

```bash
python3 -m pip install onnxruntime numpy pillow
```

Run detector:

```bash
python3 run_detector_01.py path/to/image.jpg --output detected.jpg
```

Print per-class score info:

```bash
python3 run_detector_01.py path/to/image.jpg --debug-scores
```

Lower the threshold if detections are missing:

```bash
python3 run_detector_01.py path/to/image.jpg --threshold 0.05 --output detected.jpg
```

## Mark Up A CSV

Input CSV must contain a `path` column. Relative paths are resolved relative to the CSV location.

```bash
python3 markup_document_bboxes.py input.csv --output-csv output.csv
```

The script appends a `bbox` column. For the best `document` detection, `bbox` is written as:

```json
[x1,y1,x2,y2]
```

Rows without a document detection get an empty `bbox`.

Useful options:

```bash
python3 markup_document_bboxes.py input.csv \
  --output-csv output.csv \
  --threshold 0.3 \
  --score-column document_score
```

Write all document boxes instead of only the best one:

```bash
python3 markup_document_bboxes.py input.csv --all-boxes
```
