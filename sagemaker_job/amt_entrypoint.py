#!/usr/bin/env python3
"""Container entrypoint for SageMaker automatic model tuning jobs; the image installs it as `train`.

The tuning API cannot set a container entrypoint, so SageMaker starts the image as
`docker run <image> train`. A plain ModelTrainer job overrides that with its own driver script.
A tuning job only passes the program as the `sagemaker_program` hyperparameter and leaves
running it to the image, which framework images do with sagemaker-training-toolkit.

The code is mounted as the `code` channel (sagemaker_job/tune.py adds it), the same layout
ModelTrainer jobs get, so sagemaker_job/train.py finds everything where it already looks.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HPS_PATH = Path("/opt/ml/input/config/hyperparameters.json")
DEFAULT_PROGRAM = "sagemaker_job/train.py"


def main(hps_path: Path = HPS_PATH, code_dir: Path | None = None) -> None:
    code = code_dir or Path(os.environ.get("SM_CHANNEL_CODE", "/opt/ml/input/data/code"))
    hps = json.loads(hps_path.read_text(encoding="utf-8")) if hps_path.is_file() else {}
    program = str(hps.get("sagemaker_program") or DEFAULT_PROGRAM).strip('"')
    if not (code / program).is_file():
        found = sorted(p.name for p in code.iterdir()) if code.is_dir() else "directory missing"
        sys.exit(f"train: {program} not found in {code} (contents: {found}). The tuning job needs a 'code' channel.")
    os.chdir(code)
    os.execvp(sys.executable, [sys.executable, "-u", program])


if __name__ == "__main__":
    main()
