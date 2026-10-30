"""e15b_verify.py — e15 数值验证的修复版（BUG-EXP15-01：两次调用各自消耗 rng）

正确做法：同一场景（prof0/mk/row_mod 只构造一次）下对比 dx 与 dx+p 的渲染。
验证命题：FFT 循环平移下，两假设的图像差异由且仅由非周期成分决定。
  - negative（纯周期）：差异应 ≈ 0（机器精度）→ 逐位相同 → 不可辨识（干净证据）
  - rowmod：静止行调制两帧相同 → 差异仍 ≈ 0
  - positive（非周期标记）：差异 = 标记平移的差 → 可辨识
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"
W = H = 96


def periodic_profile_pure(width: int, p: float, rng) -> np.ndarray:
    x = np.arange(width, dtype=np.float64)
    duty = rng.uniform(0.35, 0.65)
    base = ((x / p) % 1.0 < duty).astype(np.float64) * 2.0 - 1.0
    a2, a3 = rng.uniform(0.1, 0.35), rng.uniform(0.05, 0.25)
    prof = base + a2 * np.sin(2 * np.pi * x / p + rng.uniform(0, 2 * np.pi)) \
                + a3 * np.sin(4 * np.pi * x / p + rng.uniform(0, 2 * np.pi))
    return prof / (np.abs(prof).max() + 1e-9)


def fft_shift1d(prof: np.ndarray, dx: float) -> np.ndarray:
    N = len(prof)
    F = np.fft.rfft(prof)
    k = np.arange(len(F))
    return np.fft.irfft(F * np.exp(-2j * np.pi * k * dx / N), N)


def make_scene(p: float, rng, mode: str):
    """构造一次场景的全部随机成分。"""
    prof0 = periodic_profile_pure(W, p, rng)
    mk = np.zeros((H, W))
    if mode == "positive":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        for _ in range(3):
            cy, cx = rng.uniform(10, H - 10), rng.uniform(10, W - 10)
            s = rng.uniform(3, 6)
            mk += 0.5 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    row_mod = 0.08 * np.sin(2 * np.pi * np.arange(H)[:, None] / rng.uniform(30, 60))
    return prof0, mk, row_mod


def render_from_scene(scene, dx: float, mode: str) -> np.ndarray:
    prof0, mk, row_mod = scene
    prof1 = fft_shift1d(prof0, dx)
    img = prof1[None, :] + mk
    if mode == "rowmod":
        img = img + row_mod
    return ((img - img.min()) / (img.max() - img.min() + 1e-12)).astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    verify = []
    print("=== 修复版验证 max|I1(dx) - I1(dx+p)|（同一场景） ===")
    for mode in ("negative", "rowmod", "positive"):
        for p in (8.0, 16.0, 33.0, 70.0):
            diffs = []
            for _ in range(10):
                d = float(rng.uniform(0, 2 * p))
                scene = make_scene(p, rng, mode)
                Ia = render_from_scene(scene, d, mode)
                Ib = render_from_scene(scene, d + p, mode)
                diffs.append(float(np.abs(Ia - Ib).max()))
            row = {"mode": mode, "pitch": p, "max_diff": float(np.mean(diffs))}
            verify.append(row)
            print(f"  {mode:9s} p={p:4.0f}  mean max|diff| = {np.mean(diffs):.6f}")
    (OUT / "e15b_verify.json").write_text(json.dumps(verify, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e15b_verify.json'}")


if __name__ == "__main__":
    main()
