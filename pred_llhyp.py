import argparse
from pathlib import Path
from PIL import Image
import numpy as np

def load_img(path: Path) -> np.ndarray:
    """[H,W,C] float32, range [0,1]"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", required=True, help="path to a low-light image")
    parser.add_argument("-t", "--tar", required=True, help="path to a well-lit image")
    args = parser.parse_args()

    src_path = Path(args.src)
    tar_path = Path(args.tar)

    low = load_img(src_path)   # lowlight
    high = load_img(tar_path)  # well-lit

    if low.shape != high.shape:
        raise ValueError(f"shape mismatch: src={low.shape}, tar={high.shape}")

    mu_low  = low.mean()
    mu_high = high.mean()
    std_low  = low.std()
    std_high = high.std()

    # Two hyperparameters in lowlight config: Config.luminance_mean and Config.contrast_factor
    desired_mean    = mu_high
    contrast_factor = std_high / (std_low + 1e-8)

    print(f"luminance_mean (desired_mean) : {desired_mean:.6f}")
    print(f"contrast_factor               : {contrast_factor:.6f}")

if __name__ == "__main__":
    main()