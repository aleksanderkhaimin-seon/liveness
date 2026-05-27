#!/usr/bin/env python3
import ctypes.util
import os
import subprocess
import sys


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        print(f"{command[0]}: not found")
        return

    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    print(f"exit_code={completed.returncode}")


def find_library(name: str) -> None:
    print(f"{name}: {ctypes.util.find_library(name)}")


def main() -> None:
    print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

    run(["nvidia-smi"])
    run(["ldconfig", "-p"])

    print("\nLibrary lookup:")
    for library in ("cuda", "cudart", "cublas", "cudnn", "cufft", "curand"):
        find_library(library)

    print("\nTensorFlow:")
    import tensorflow as tf

    print("tf.__version__:", tf.__version__)
    print("built with cuda:", tf.test.is_built_with_cuda())
    print("physical GPUs:", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()
