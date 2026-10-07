"""Progress reporting shared by the scoring scripts; no TensorFlow import."""
from __future__ import annotations

import sys


def progress(iterable, total: int, desc: str):
    """tqdm bar on a terminal; a plain line roughly every 10% otherwise (SageMaker logs don't redraw bars)."""
    try:
        from tqdm import tqdm
        if sys.stdout.isatty():
            yield from tqdm(iterable, total=total, desc=desc)
            return
    except ImportError:
        pass
    every = max(1, total // 10)
    print(desc, flush=True)
    for i, item in enumerate(iterable, start=1):
        yield item
        if i % every == 0 or i == total:
            print(f"  {i}/{total} ({100 * i // max(total, 1)}%)", flush=True)
