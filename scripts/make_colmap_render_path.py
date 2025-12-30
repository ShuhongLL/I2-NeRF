#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 LLFF dataloader 实际使用的 camtoworlds 生成插值后的 render_path poses。

关键点：
- 直接用 internal.datasets.LLFF + internal.configs.Config，
  复用你训练 / 测试时的那套 pose 管线；
- config.render_path=False：保证 ds.camtoworlds = poses（而不是内部生成的 render_poses）；
- 在这些 poses 上做四元数 + 线性插值，输出 [N_render, 3, 4] float32；
- 输出文件可以直接设为 Config.render_path_file。

Usage:

python scripts/make_llff_render_path_from_dataset.py \
    --data_dir /path/to/scene \
    --out_path /path/to/scene/render_poses.npy \
    --steps_between 15 \
    --no_loop \
    --forward_facing   # 如果你训练时 forward_facing=True，就加这个
"""

import argparse
from pathlib import Path

import numpy as np

from internal import configs
from internal import datasets as dataset_lib
from internal import camera_utils


# ---------- 四元数插值工具 ----------

def mat3_to_quat(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def quat_to_mat3(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = (1.0 - t) * q0 + t * q1
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def interpolate_c2w(c2w_4x4: np.ndarray,
                    steps_between: int = 10,
                    loop: bool = True) -> np.ndarray:
    assert c2w_4x4.ndim == 3 and c2w_4x4.shape[1:] == (4, 4)
    N = c2w_4x4.shape[0]

    R_all = c2w_4x4[:, :3, :3]
    t_all = c2w_4x4[:, :3, 3]
    q_all = np.stack([mat3_to_quat(R) for R in R_all], axis=0)

    out = []
    num_segments = N if loop else (N - 1)

    for i in range(num_segments):
        j = (i + 1) % N
        q0, q1 = q_all[i], q_all[j]
        t0, t1 = t_all[i], t_all[j]

        if i == 0:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_all[i]
            pose[:3, 3] = t0
            out.append(pose[:3, :4])

        for s in range(1, steps_between + 1):
            alpha = s / float(steps_between + 1)
            q = slerp(q0, q1, alpha)
            R = quat_to_mat3(q)
            t = (1.0 - alpha) * t0 + alpha * t1

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, 3] = t
            out.append(pose[:3, :4])

    return np.stack(out, axis=0)


def main():
    parser = argparse.ArgumentParser("Make render_path from LLFF dataset poses.")
    parser.add_argument("--data_dir", "-d", type=str, required=True,
                        help="LLFF 场景根目录 (包含 images/, sparse/0/)")
    parser.add_argument("--out_path", "-o", type=str, required=True,
                        help="输出 .npy (Config.render_path_file)")
    parser.add_argument("--steps_between", type=int, default=10,
                        help="相邻两帧之间插值的帧数（不含端点）")
    parser.add_argument("--no_loop", action="store_true",
                        help="不在最后一帧和第一帧之间闭环")
    parser.add_argument("--forward_facing", action="store_true",
                        help="必须和训练时的 config.forward_facing 一致")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path)
    loop = not args.no_loop

    # 1. 构造一个 Config，复用 LLFF loader
    cfg = configs.Config()
    cfg.dataset_loader = "llff"
    cfg.data_dir = str(data_dir)

    # 这些是 Dataset.__init__ 里需要的最小字段
    cfg.patch_size = 1
    cfg.batch_size = 1024
    cfg.world_size = 1
    cfg.global_rank = 1
    cfg.batching = "all_images"
    cfg.use_tiffs = False
    cfg.compute_disp_metrics = False
    cfg.compute_normal_metrics = False
    cfg.num_border_pixels_to_mask = 0
    cfg.apply_bayer_mask = False
    cfg.enable_bcp = False
    cfg.enable_depth_prior = False
    cfg.use_bcp_atmospheric_light = False
    cfg.compute_visibility = False

    # 和你训练时真正会影响 pose 的参数：
    cfg.forward_facing = args.forward_facing
    cfg.render_path = False          # 非常关键！确保 camtoworlds = poses
    cfg.factor = 0                   # 下采样只影响 intrinsics，不影响 extrinsics
    cfg.llff_use_all_images_for_training = True
    cfg.llff_use_all_images_for_testing = True
    cfg.llffhold = 8                 # 随便设，反正上面已经要求 use_all_images

    # 2. 实例化 LLFF 数据集
    ds = dataset_lib.LLFF(split="train", data_dir=str(data_dir), config=cfg)

    # 此时 ds.camtoworlds 就是 LLFF loader 最终使用的 poses（在 NeRF 世界坐标系里）
    poses = ds.camtoworlds           # [N, 3, 4]
    print("[INFO] LLFF camtoworlds shape:", poses.shape)

    # 3. 拼成 4x4，做插值
    c2w_4x4 = np.tile(np.eye(4, dtype=np.float32)[None, ...],
                      (poses.shape[0], 1, 1))
    c2w_4x4[:, :3, :4] = poses

    render_c2w_3x4 = interpolate_c2w(
        c2w_4x4,
        steps_between=args.steps_between,
        loop=loop,
    )
    print("[INFO] render path shape:", render_c2w_3x4.shape)

    # 4. 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, render_c2w_3x4.astype(np.float32))
    print(f"[OK] saved render path to {out_path}")


if __name__ == "__main__":
    main()
