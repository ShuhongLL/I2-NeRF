#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an interpolated camera path for Blender-style datasets
(transforms_*.json) to be used as Config.render_path_file in zipnerf-pytorch.

Usage example:
    python scripts/make_blender_render_path.py \
        --transforms_path /path/to/lego/transforms_test.json \
        --out_path /path/to/lego_render_poses.npy \
        --steps_between 10 \
        --no_loop
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_blender_c2w(transforms_path: Path) -> np.ndarray:
    """Load [N, 4, 4] cam2world matrices from transforms_*.json."""
    with open(transforms_path, "r") as f:
        meta = json.load(f)

    mats = []
    for frame in meta["frames"]:
        mat = np.array(frame["transform_matrix"], dtype=np.float32)
        mats.append(mat)
    return np.stack(mats, axis=0)  # [N, 4, 4]


# ---------- basic quaternion utils (w, x, y, z) ----------

def mat3_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z)."""
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
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
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
    """Spherical linear interpolation between two unit quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))

    # Avoid taking the long way round
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Nearly identical: fall back to linear interpolation
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


def interpolate_c2w(
    c2w_4x4: np.ndarray,
    steps_between: int = 10,
    loop: bool = True,
) -> np.ndarray:
    """
    给定 [N, 4, 4] 的 cam2world, 在相邻相机之间插值，生成更 dense 的路径。

    steps_between: 每两个 keyframe 之间插入多少个中间帧（不含端点）。
    loop: 是否最后一个 camera 和第一个 camera 之间也插值一圈。
    """
    assert c2w_4x4.ndim == 3 and c2w_4x4.shape[1:] == (4, 4)
    N = c2w_4x4.shape[0]

    R_all = c2w_4x4[:, :3, :3]
    t_all = c2w_4x4[:, :3, 3]
    q_all = np.stack([mat3_to_quat(R) for R in R_all], axis=0)

    out = []

    # segment 数：loop 用 N 段（包括最后 -> 第一个），否则 N-1 段
    num_segments = N if loop else (N - 1)

    for i in range(num_segments):
        j = (i + 1) % N
        q0, q1 = q_all[i], q_all[j]
        t0, t1 = t_all[i], t_all[j]

        # 每段的起点：只在第一段时加入，避免重复
        if i == 0:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_all[i]
            pose[:3, 3] = t0
            out.append(pose[:3, :4])  # 存成 3x4 即可

        # 中间的插值帧
        for s in range(1, steps_between + 1):
            alpha = s / float(steps_between + 1)
            q = slerp(q0, q1, alpha)
            R = quat_to_mat3(q)
            t = (1.0 - alpha) * t0 + alpha * t1

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, 3] = t
            out.append(pose[:3, :4])

    return np.stack(out, axis=0)  # [N_render, 3, 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transforms_path",
        "-i",
        type=str,
        required=True,
        help="Path to transforms_*.json (e.g. transforms_test.json).",
    )
    parser.add_argument(
        "--out_path",
        "-o",
        type=str,
        required=True,
        help="Output .npy file to be used as Config.render_path_file.",
    )
    parser.add_argument(
        "--steps_between",
        type=int,
        default=10,
        help="Number of interpolated poses between each pair of input cameras.",
    )
    parser.add_argument(
        "--no_loop",
        action="store_true",
        help="If set, do NOT connect last camera back to the first.",
    )
    args = parser.parse_args()

    transforms_path = Path(args.transforms_path)
    out_path = Path(args.out_path)
    loop = not args.no_loop

    c2w_4x4 = load_blender_c2w(transforms_path)
    render_c2w_3x4 = interpolate_c2w(
        c2w_4x4,
        steps_between=args.steps_between,
        loop=loop,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, render_c2w_3x4.astype(np.float32))
    print(f"[OK] Saved render path of shape {render_c2w_3x4.shape} to {out_path}")


if __name__ == "__main__":
    main()
