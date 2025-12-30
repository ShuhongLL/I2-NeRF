#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
from natsort import natsorted  # 如果没有这个库，可以改成内置 sorted

def parse_args():
    parser = argparse.ArgumentParser("Make video from images with given prefix")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="Folder containing images"
    )
    parser.add_argument(
        "--prefix", "-p", required=True,
        help="Prefix of images to use (filename startswith this)"
    )
    parser.add_argument(
        "--fps", type=float, required=True,
        help="Frames per second of output video"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="(Optional) Output video path, default: <input_path>/<prefix>_video.mp4"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_path
    prefix = args.prefix
    fps = args.fps

    # 允许的图片后缀
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    # 收集匹配前缀的图片
    all_files = os.listdir(input_dir)
    img_files = [
        f for f in all_files
        if f.startswith(prefix) and f.endswith(exts)
    ]

    if len(img_files) == 0:
        raise RuntimeError(f"No images found in {input_dir} with prefix '{prefix}'")

    # 自然排序（按数字顺序），例如 1,2,10 而不是 1,10,2
    try:
        img_files = natsorted(img_files)
    except Exception:
        img_files = sorted(img_files)

    img_paths = [os.path.join(input_dir, f) for f in img_files]

    print(f"[INFO] Found {len(img_paths)} images.")
    print("\n".join(img_paths[:5]))
    if len(img_paths) > 5:
        print("...")

    # 读第一张图获取尺寸
    first_img = cv2.imread(img_paths[0])
    if first_img is None:
        raise RuntimeError(f"Failed to read first image: {img_paths[0]}")

    height, width = first_img.shape[:2]
    size = (width, height)

    # 输出路径
    if args.output is None:
        out_path = os.path.join(input_dir, f"{prefix}_video.mp4")
    else:
        out_path = args.output

    # 使用 mp4v 编码
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    for idx, path in enumerate(img_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Failed to read image: {path}, skip.")
            continue

        h, w = img.shape[:2]
        if (w, h) != size:
            # 尺寸不一致时自动 resize
            img = cv2.resize(img, size)
        writer.write(img)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(img_paths):
            print(f"[INFO] Written {idx + 1}/{len(img_paths)} frames.")

    writer.release()
    print(f"[DONE] Saved video to: {out_path}")


if __name__ == "__main__":
    main()
