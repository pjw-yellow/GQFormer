#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def read_split_list(txt_path: Path):
    names = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            names.append(Path(s).stem)
    return names


def ensure_dirs(out_root: Path):
    for sub in [
        "img/train", "img/val",
        "ann/train", "ann/val",
        "ann_color/train", "ann_color/val",
    ]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)


def build_category_mapping():
    category_mapping = {
        0: (128, 64, 128),
        1: (244, 35, 232),
        2: (70, 70, 70),
        3: (192, 0, 128),
        4: (190, 153, 153),
        5: (153, 153, 153),
        6: (250, 170, 30),
        7: (220, 220, 0),
        8: (107, 142, 35),
        9: (152, 251, 152),
        10: (70, 130, 180),
        11: (220, 20, 60),
        12: (230, 149, 139),
        13: (0, 0, 142),
        14: (0, 0, 70),
        15: (90, 40, 40),
        16: (0, 80, 100),
        17: (0, 253, 253),
        18: (0, 68, 62),
        255: (0, 0, 0),
    }
    return category_mapping


def colorize_mask(mask, mapping):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in mapping.items():
        rgb[mask == k] = v
    return rgb


def transfer(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def process_split(split, names, src_img_dir, src_mask_dir, out_root, mapping, mode):
    for name in names:
        img_src = src_img_dir / f"{name}.jpg"
        mask_src = src_mask_dir / f"{name}.png"

        if not img_src.exists() or not mask_src.exists():
            continue

        img_dst = out_root / "img" / split / img_src.name
        mask_dst = out_root / "ann" / split / mask_src.name
        color_dst = out_root / "ann_color" / split / mask_src.name

        transfer(img_src, img_dst, mode)
        transfer(mask_src, mask_dst, mode)

        mask = np.array(Image.open(mask_dst if mode == "move" else mask_src))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        color = colorize_mask(mask.astype(np.uint8), mapping)
        Image.fromarray(color).save(color_dst)


def main():
    parser = argparse.ArgumentParser()

    # Path to the original RailSem19 root directory
    parser.add_argument(
        "--src_root",
        type=str,
        default="path/to/railsem19"
    )

    # Output directory to store img / ann / ann_color
    parser.add_argument(
        "--out_root",
        type=str,
        default="path/to/myr19pros"
    )

    # File transfer mode: copy (safe) or move (will remove originals)
    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "move"]
    )

    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    ensure_dirs(out_root)

    splits_dir = src_root / "rs19_splits4000"
    img_dir = src_root / "rs19_val" / "jpgs" / "rs19_val"
    mask_dir = src_root / "rs19_val" / "uint8" / "rs19_val"

    train_list = read_split_list(splits_dir / "train.txt")
    val_list = read_split_list(splits_dir / "val.txt")

    mapping = build_category_mapping()

    process_split("train", train_list, img_dir, mask_dir, out_root, mapping, args.mode)
    process_split("val", val_list, img_dir, mask_dir, out_root, mapping, args.mode)


if __name__ == "__main__":
    main()
