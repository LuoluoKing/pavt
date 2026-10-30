"""e6_efficiency.py — E6：CPU 单帧延迟与内存（真实计时）

测三档：
  1. autocorrelation only —— 2D 局部自相关估计节距（96×96 patch）
  2. phase correlation   —— 一对 patch 的互相关 + 抛物线亚像素
  3. PAVT (hybrid)       —— 2 + 特征提取 + 可观测性网络前向
"""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

import e1_baselines as B
from e1_hybrid import features, ObsNet

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"


def acf_pitch(img: np.ndarray) -> float:
    """2D 自相关 + 径向平均，取第一个显著局部极大。"""
    x = img - img.mean()
    F = np.fft.rfft2(x, s=(2 * x.shape[0], 2 * x.shape[1]))
    acf = np.fft.irfft2(F * np.conj(F), s=(2 * x.shape[0], 2 * x.shape[1]))
    acf = acf[: x.shape[0], : x.shape[1]]
    acf /= acf[0, 0] + 1e-12
    h, w = acf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy, xx).astype(int)
    prof = np.bincount(r.ravel(), weights=acf.ravel()) / (np.bincount(r.ravel()) + 1e-9)
    for i in range(2, min(len(prof) - 2, 60)):
        if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > 0.1:
            return float(i)
    return float(len(prof) - 1)


def bench(fn, n=150):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    rng = np.random.default_rng(0)
    p = 24.0
    I0, I1 = B.render(p, 3.0, rng)

    model = ObsNet(71)
    model.eval()
    with torch.no_grad():
        f, phi, ratio = features(I0, I1, p)
        _ = model(torch.from_numpy(f)[None])

    t_acf = bench(lambda: acf_pitch(I0))
    t_pc = bench(lambda: B.disp_band(I0, I1, p))

    def hybrid():
        f, _, _ = features(I0, I1, p)
        with torch.no_grad():
            model(torch.from_numpy(f)[None])

    t_hy = bench(hybrid)

    def mem(fn):
        tracemalloc.start()
        fn()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 2**20

    rows = [
        {"method": "Autocorrelation only", "latency_ms": t_acf, "mem_mb": mem(lambda: acf_pitch(I0))},
        {"method": "Phase correlation", "latency_ms": t_pc, "mem_mb": mem(lambda: B.disp_band(I0, I1, p))},
        {"method": "PAVT (hybrid)", "latency_ms": t_hy, "mem_mb": mem(hybrid)},
    ]
    for r in rows:
        print(f"{r['method']:22s} {r['latency_ms']:7.2f} ms  {r['mem_mb']:6.2f} MB")
    (OUT / "e6_efficiency.json").write_text(
        json.dumps({"patch": 96, "n_repeat": 150, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"已写入 {OUT/'e6_efficiency.json'}")


if __name__ == "__main__":
    main()
