"""e12_unfolded_and_wrap.py — ⑥ 的两个定案子项

(1) 未折叠曲面：把选择器的输入从 ±p/2 扩到 ±2p 重训，看 AURC 是否改善。
    若显著改善 → 现有设计（折叠）确实构成天花板；若不改善 → 天花板不在输入带宽。
(2) wrap 错位：对全部 16 个 pitch，用**分数位移**比较 nd_shift(wrap) 与 FFT 环移
    下 Δ 与 Δ+p 的差异，确认可公度只在 p|W 时成立。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import shift as nd_shift

import e1_baselines as B
from e1_hybrid import NBIN, ObsNet, coverage_curve
from e7_head_ablation import train as train_head

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
SEED = 20261010
W = 96


def features_band(I0, I1, p, band_mult=2.0):
    """同 e1_hybrid.features，但曲面窗口为 ±band_mult*p。"""
    c, lags = B.corr_surface(I0, I1)
    lim = band_mult * p
    m = np.abs(lags) <= lim
    cc, ll = c[m], lags[m]
    i = int(np.argmax(cc))
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-12)
        phi = float(ll[i] + 0.5 * d)
    else:
        phi = float(ll[i])
    xs = np.linspace(-lim, lim, NBIN)
    surf = np.interp(xs, ll, cc)
    peak = float(cc.max())
    mask = np.ones_like(cc, dtype=bool)
    mask[max(0, i - 2):i + 3] = False
    second = float(cc[mask].max()) if mask.any() else float(cc.min())
    ratio = (peak - second) / (abs(peak) + 1e-6)
    hf = float(np.std(I0 - np.median(I0, axis=0, keepdims=True)))
    colstd = float(np.std(I0.mean(axis=0)))
    extra = np.array([p, peak, ratio, second, hf, colstd, abs(phi) / p], dtype=np.float32)
    return np.concatenate([surf.astype(np.float32), extra]), phi


def build_band(n_per_cell=10, band_mult=2.0):
    rng = np.random.default_rng(SEED)
    degs = [("none", {}), ("noise20", {"noise": 20}), ("noise60", {"noise": 60}),
            ("occl15", {"occl": 0.15}), ("occl30", {"occl": 0.30}), ("jpeg30", {"jpeg_q": 30})]
    X, y = [], []
    for regime in ("small", "large"):
        for p in B.PITCHES:
            lo, hi = (0.0, p / 2.0) if regime == "small" else (p / 2.0, 2.0 * p)
            for _, deg in degs:
                for _ in range(n_per_cell):
                    d = float(rng.uniform(lo, hi))
                    I0, I1 = B.render(p, d, rng, **deg)
                    f, phi = features_band(I0, I1, p, band_mult)
                    X.append(f)
                    y.append(abs(phi - d) / p)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


def main() -> None:
    print("=== (2) wrap 错位：全 16 pitch，分数位移 ===")
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'p':>4} {'96%p':>5} {'nd_shift max|Δ-(Δ+p)|':>22} {'FFT':>8}")
    for p in B.PITCHES:
        prof = B.periodic_profile(W, p, rng)
        d0 = 0.37 * p
        a = nd_shift(prof, d0, order=3, mode="wrap")
        b = nd_shift(prof, d0 + p, order=3, mode="wrap")
        dn = float(np.abs(a - b).max())
        k = np.fft.rfftfreq(W)
        fa = np.fft.irfft(np.fft.rfft(prof) * np.exp(-2j * np.pi * k * d0), n=W)
        fb = np.fft.irfft(np.fft.rfft(prof) * np.exp(-2j * np.pi * k * (d0 + p)), n=W)
        df = float(np.abs(fa - fb).max())
        rows.append({"p": p, "mod": W % p, "nd": dn, "fft": df})
        print(f"{p:4.0f} {W%p:5d} {dn:22.4f} {df:8.4f}")
    out = {"wrap": rows}

    # (1) 未折叠曲面重训
    print("\n=== (1) 未折叠曲面（±2p）重训 ===")
    X2, y2 = build_band(10, 2.0)
    idx = np.random.default_rng(SEED).permutation(len(X2))
    ntr = int(0.7 * len(X2))
    tr, te = idx[:ntr], idx[ntr:]
    torch.manual_seed(SEED)
    m = train_head(ObsNet(X2.shape[1]), X2[tr], y2[tr])
    with torch.no_grad():
        s = m(torch.from_numpy(X2[te])).numpy()
    r = coverage_curve(s, y2[te])
    print(f"  ±2p 曲面：AURC {r['aurc']:.4f}  MAE@60% {r[0.6]:.4f}  (对照：折叠 ±p/2 为 0.354)")
    out["unfolded"] = r

    (OUT / "e12_unfolded_wrap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(f"\n已写入 {OUT/'e12_unfolded_wrap.json'}")


if __name__ == "__main__":
    main()
