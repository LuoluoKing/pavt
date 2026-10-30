"""e1_baselines.py — E1 阶段一：经典方法在合成周期结构上的谐波歧义测量

目的
----
在**完全可控**的合成数据上，测出经典估计器的两个量：
  · 位移误差 MAE（像素）
  · 谐波歧义率 HAR（分支判错的比例）
并按两种运动幅度分别报告：
  · 小位移  |Δ| < p/2  （帧间典型）
  · 大位移  |Δ| ∈ [p/2, 2p) （快速运动，分支不可忽略）

方法
----
1. global-corr : 全范围互相关 argmax（无先验 → 暴露歧义）
2. band-corr   : 仅在 [−p/2, p/2) 内搜索（有先验 → 结构上不可能跳支）
3. lk-1d       : 行平均剖面上的 1D Lucas–Kanade（梯度法），结果按 p 折叠

数据
----
1D 周期结构（方波 + 谐波 + 抖动），96×96，随机相位，刚性平移 Δ。
退化：none / noise σ=20,60 / occlusion 0.15 / jpeg q=30。

输出
----
04-实验/results/e1_baselines.json  +  控制台汇总表
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as nd_shift

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
OUT.mkdir(parents=True, exist_ok=True)

PITCHES = [8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 44, 50, 56, 60, 70, 80]
N_PER_CELL = 24
SEED = 20261010
H = W = 96


# ----------------------------------------------------------------- 数据生成
def periodic_profile(width: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """周期为 p 的 1D 剖面：方波 + 2/3 次谐波 + 每样本抖动，避免退化为纯正弦。"""
    x = np.arange(width, dtype=np.float64)
    duty = rng.uniform(0.35, 0.65)
    base = ((x / p) % 1.0 < duty).astype(np.float64) * 2.0 - 1.0
    a2, a3 = rng.uniform(0.1, 0.35), rng.uniform(0.05, 0.25)
    prof = base + a2 * np.sin(2 * np.pi * x / p + rng.uniform(0, 2 * np.pi)) \
                + a3 * np.sin(4 * np.pi * x / p + rng.uniform(0, 2 * np.pi))
    # 轻微的沿 x 的亮度渐变（模拟光照），不影响周期性
    prof = prof + 0.15 * (x / width - 0.5)
    return prof / (np.abs(prof).max() + 1e-9)


def render(p: float, dx: float, rng: np.random.Generator,
           noise: float = 0.0, occl: float = 0.0, jpeg_q: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """返回 (I0, I1)，I1 为 I0 沿 x 平移 dx 像素（刚性）。"""
    prof0 = periodic_profile(W, p, rng)
    # 2D：每行叠加轻微纹理，使图像不是简单的列复制
    row_mod = 0.08 * np.sin(2 * np.pi * np.arange(H)[:, None] / rng.uniform(30, 60))
    I0 = prof0[None, :] + row_mod
    I0 = (I0 - I0.min()) / (I0.max() - I0.min())

    # 精确平移：先用 3 次样条平移剖面，再广播（保持刚性、避免边界插值伪影）
    prof1 = nd_shift(prof0, dx, order=3, mode="wrap")
    I1 = prof1[None, :] + row_mod
    I1 = (I1 - I0.min()) / (I0.max() - I0.min())

    out = []
    for I in (I0, I1):
        img = I.copy()
        if noise > 0:
            img = img + rng.normal(0, noise / 255.0, img.shape)
        if occl > 0:
            w = int(round(W * occl))
            x0 = rng.integers(0, max(1, W - w))
            img[:, x0:x0 + w] = img.mean()
        out.append(img)
    if jpeg_q:
        from PIL import Image
        enc = []
        for img in out:
            a = np.clip(img * 255, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, format="JPEG", quality=jpeg_q)
            buf.seek(0)
            enc.append(np.asarray(Image.open(buf), dtype=np.float64) / 255.0)
        out = enc
    return out[0], out[1]


# ------------------------------------------------------------------- 估计器
def corr_surface(I0: np.ndarray, I1: np.ndarray) -> np.ndarray:
    """沿 x 的归一化互相关曲面（列剖面 + Hann 窗）。"""
    a = I0.mean(axis=0)
    b = I1.mean(axis=0)
    a = (a - a.mean()) * np.hanning(len(a))
    b = (b - b.mean()) * np.hanning(len(b))
    A, B = np.fft.rfft(a, n=2 * len(a)), np.fft.rfft(b, n=2 * len(b))
    c = np.fft.irfft(B * np.conj(A), n=2 * len(a))
    c = np.concatenate([c[-len(a) + 1:], c[:len(a)]])   # 滞后轴 [-W+1, W-1]
    lags = np.arange(-len(a) + 1, len(a))
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return c / denom, lags


def disp_global(I0, I1):
    c, lags = corr_surface(I0, I1)
    return float(lags[int(np.argmax(c))])


def disp_band(I0, I1, p):
    c, lags = corr_surface(I0, I1)
    m = np.abs(lags) <= p / 2.0
    cc, ll = c[m], lags[m]
    i = int(np.argmax(cc))
    # 抛物线插值
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-12)
        return float(ll[i] + 0.5 * d)
    return float(ll[i])


def disp_lk1d(I0: np.ndarray, I1: np.ndarray, p: float, iters: int = 15, sig: float = 1.5):
    """标准 1D Lucas–Kanade（单尺度、高斯平滑、无阻尼）。

    约定：I1(x) = I0(x − d)。以 shift(b, −d) 对齐后最小化残差，
    更新 d ← d − <g, r>/<g, g>。**注意 shift 方向**：早期版本写成
    shift(b, +d) 会让迭代发散到 ±p/2（已修正）。
    """
    from scipy.ndimage import gaussian_filter1d
    a = gaussian_filter1d(I0.mean(axis=0), sig, mode="wrap")
    b = gaussian_filter1d(I1.mean(axis=0), sig, mode="wrap")
    d = 0.0
    for _ in range(iters):
        bs = nd_shift(b, -d, order=3, mode="wrap")
        gx = np.gradient((a + bs) / 2.0)
        r = bs - a
        step = float(np.sum(gx * r) / (np.sum(gx * gx) + 1e-9))
        d -= step
        if abs(step) < 1e-4:
            break
    return float(d)


# ----------------------------------------------------------------- 指标
def branch(x: float, p: float) -> int:
    return int(np.round(x / p))


def run_cell(p: float, regime: str, deg: dict, rng: np.random.Generator) -> dict:
    lo, hi = (0.0, p / 2.0) if regime == "small" else (p / 2.0, 2.0 * p)
    errs = {k: [] for k in ("global", "band", "lk1d")}
    har = {k: 0 for k in ("global", "band", "lk1d")}
    n = 0
    for _ in range(N_PER_CELL):
        dx_true = float(rng.uniform(lo, hi))
        I0, I1 = render(p, dx_true, rng, **deg)
        kb = branch(dx_true, p)
        for name, est in (("global", disp_global(I0, I1)),
                          ("band", disp_band(I0, I1, p)),
                          ("lk1d", disp_lk1d(I0, I1, p))):
            errs[name].append(abs(est - dx_true))
            har[name] += int(branch(est, p) != kb)
        n += 1
    return {
        "pitch": p, "regime": regime, "degradation": deg,
        "n": n,
        "mae": {k: float(np.mean(v)) for k, v in errs.items()},
        "har": {k: v / n for k, v in har.items()},
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    degs = [
        ("none", {}),
        ("noise20", {"noise": 20}),
        ("noise60", {"noise": 60}),
        ("occl15", {"occl": 0.15}),
        ("jpeg30", {"jpeg_q": 30}),
    ]
    cells = []
    for regime in ("small", "large"):
        for p in PITCHES:
            for dname, deg in degs:
                cells.append(run_cell(p, regime, deg, rng))

    # 汇总
    def agg(regime, dname, key):
        sel = [c for c in cells if c["regime"] == regime and
               c["degradation"] == ({} if dname == "none" else dict(degs)[dname])]
        if not sel:
            sel = [c for c in cells if c["regime"] == regime]
        return {m: float(np.mean([c[key][m] for c in sel])) for m in ("global", "band", "lk1d")}

    print(f"{'regime':6s} {'degradation':10s} | {'MAE global':>10s} {'MAE band':>9s} {'MAE lk1d':>9s} "
          f"| {'HAR glob':>8s} {'HAR band':>8s} {'HAR lk1d':>8s}")
    for regime in ("small", "large"):
        for dname, deg in degs:
            sel = [c for c in cells if c["regime"] == regime and c["degradation"] == deg]
            mae = {m: np.mean([c["mae"][m] for c in sel]) for m in ("global", "band", "lk1d")}
            har = {m: np.mean([c["har"][m] for c in sel]) for m in ("global", "band", "lk1d")}
            print(f"{regime:6s} {dname:10s} | {mae['global']:10.2f} {mae['band']:9.2f} {mae['lk1d']:9.2f} "
                  f"| {har['global']:8.3f} {har['band']:8.3f} {har['lk1d']:8.3f}")

    (OUT / "e1_baselines.json").write_text(
        json.dumps({"seed": SEED, "n_per_cell": N_PER_CELL, "pitches": PITCHES,
                    "cells": cells}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已写入 {OUT/'e1_baselines.json'}（{len(cells)} 个格点）")


if __name__ == "__main__":
    main()
