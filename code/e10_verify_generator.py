"""e10_verify_generator.py — 验证两件决定性事实

(1) nd_shift(order=3, mode='wrap') 是否真环移？比较 Δ 与 Δ+p 两幅渲染图的
    max|diff|，与 FFT 相位位移版本对照。若插值伪影 ~1，则"分支不可辨识"
    在自有数据里不成立（伪影本身即信息）。
(2) M9：曲面自归一化 vs 峰高（真跑并落盘）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as nd_shift

import e1_baselines as B
from e1_hybrid import ObsNet, build, coverage_curve
from e7_head_ablation import train as train_head

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
SEED = 20261010


def circ_shift(v: np.ndarray, d: float) -> np.ndarray:
    k = np.fft.rfftfreq(len(v))
    return np.fft.irfft(np.fft.rfft(v) * np.exp(-2j * np.pi * k * d), n=len(v))


def main() -> None:
    print("=== (1) 生成器是否真环移 ===")
    rng = np.random.default_rng(0)
    for p in (8.0, 16.0, 32.0):
        prof = B.periodic_profile(96, p, rng)
        base = prof.copy()
        # 当前生成器的做法
        a = nd_shift(base, 0.0, order=3, mode="wrap")
        b = nd_shift(base, p, order=3, mode="wrap")
        d_nd = float(np.abs(a - b).max())
        # FFT 环移
        a2 = circ_shift(base, 0.0)
        b2 = circ_shift(base, p)
        d_fft = float(np.abs(a2 - b2).max())
        print(f"  p={p:4.0f}: nd_shift max|Δ-(Δ+p)| = {d_nd:.3e} | FFT = {d_fft:.3e} "
              f"| 纹理 std = {base.std():.3f} | noise20 = {20/255:.3f}")

    print("\n=== (2) M9：曲面归一化对照（真跑并落盘）===")
    X, y, meta = build()
    Xn = X.copy()
    for i in range(len(Xn)):
        s = Xn[i, :64]
        m = np.abs(s).max()
        if m > 1e-6:
            Xn[i, :64] = s / m

    idx = np.random.default_rng(SEED).permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    res = {}
    for name, Xin in (("raw surface", X), ("max-normalised surface", Xn)):
        torch_model = ObsNet(Xin.shape[1])
        import torch
        torch.manual_seed(SEED)
        m = train_head(torch_model, Xin[tr], y[tr])
        with torch.no_grad():
            s = m(torch.from_numpy(Xin[te])).numpy()
        res[name] = coverage_curve(s, y[te])
        print(f"  {name:24s} AURC {res[name]['aurc']:.4f}  MAE@60% {res[name][0.6]:.4f}")

    import torch
    peak = X[te, 65]
    res["peak height only"] = coverage_curve(-peak, y[te])
    print(f"  {'peak height only':24s} AURC {res['peak height only']['aurc']:.4f}  "
          f"MAE@60% {res['peak height only'][0.6]:.4f}")

    (OUT / "e10_generator_and_m9.json").write_text(
        json.dumps({"m9": res, "note": "peak-only 用留出集，与其他行同口径"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e10_generator_and_m9.json'}")


if __name__ == "__main__":
    main()
