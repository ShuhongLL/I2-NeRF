import os
import glob
import argparse

import numpy as np
from PIL import Image
from natsort import natsorted
from transformers import pipeline


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Min-max normalize a depth-like map to [0, 1]."""
    depth = depth.astype(np.float32)
    dmin = float(depth.min())
    dmax = float(depth.max())
    denom = dmax - dmin
    if denom < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - dmin) / (denom + 1e-8)


def is_image_file(path: str) -> bool:
    """Return True if the file extension looks like an image."""
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Depth Anything V2 predicted_depth and save as 8-bit PNG depth maps."
    )
    parser.add_argument(
        "-s", "--source_root",
        required=True,
        help="Root directory containing scene subfolders (e.g., /path/to/LOM)."
    )
    parser.add_argument(
        "-n", "--name",
        required=True,
        help='Image folder name inside each scene (e.g., "low" for LOM, "images" for SeaThruNeRF).'
    )
    parser.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help='Hugging Face model id for depth-estimation pipeline.'
    )
    parser.add_argument(
        "--out_folder",
        default="depth",
        help='Output folder name inside each scene (default: "depth").'
    )
    args = parser.parse_args()

    source_root = args.source_root
    name = args.name

    # Create depth pipeline once.
    pipe = pipeline(task="depth-estimation", model=args.model)

    # Enumerate scene directories.
    scene_dirs = [
        d for d in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, d))
    ]
    scene_dirs = natsorted(scene_dirs)

    for scene in scene_dirs:
        image_dir = os.path.join(source_root, scene, name)
        out_dir = os.path.join(source_root, scene, args.out_folder)
        os.makedirs(out_dir, exist_ok=True)

        # Collect all files under image_dir (filter to images).
        image_paths = glob.glob(os.path.join(image_dir, "*"))
        image_paths = [p for p in image_paths if os.path.isfile(p) and is_image_file(p)]
        image_paths = natsorted(image_paths)

        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")

            # Use the real-valued output, not the visualization output.
            pred = pipe(image)["predicted_depth"]  # typically a torch tensor
            try:
                depth_np = pred.squeeze().detach().cpu().numpy()
            except Exception:
                # Fallback if the returned object is already array-like
                depth_np = np.array(pred).squeeze()

            depth_np = normalize_depth(depth_np)

            depth_np = 1.0 - depth_np

            depth_uint8 = (depth_np * 255.0).clip(0, 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_uint8, mode="L")

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(out_dir, f"{base_name}.png")
            depth_img.save(output_path)

        print(f"Finish processing {scene}")


if __name__ == "__main__":
    main()
