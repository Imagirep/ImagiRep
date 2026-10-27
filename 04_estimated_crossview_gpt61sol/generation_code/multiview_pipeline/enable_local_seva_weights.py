#!/usr/bin/env python3
"""Let the pinned SEVA checkout use locally mirrored VAE and CLIP weights."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        print(f"already patched: {path}")
        return
    if source.count(old) != 1:
        raise ValueError(f"Expected one match in {path}; found {source.count(old)}")
    path.write_text(source.replace(old, new))
    print(f"patched: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seva-root", type=Path, required=True)
    args = parser.parse_args()
    replace_once(
        args.seva_root / "seva/modules/autoencoder.py",
        'import torch\n',
        'import os\n\nimport torch\n',
    )
    replace_once(
        args.seva_root / "seva/modules/autoencoder.py",
        '            "stabilityai/stable-diffusion-2-1-base",\n',
        '            os.environ.get("SEVA_VAE_ROOT", "stabilityai/stable-diffusion-2-1-base"),\n',
    )
    replace_once(
        args.seva_root / "seva/modules/conditioner.py",
        'import kornia\n',
        'import os\n\nimport kornia\n',
    )
    replace_once(
        args.seva_root / "seva/modules/conditioner.py",
        '            "ViT-H-14", pretrained="laion2b_s32b_b79k"\n',
        '            "ViT-H-14", pretrained=os.environ.get("SEVA_OPEN_CLIP_WEIGHTS", "laion2b_s32b_b79k")\n',
    )


if __name__ == "__main__":
    main()
