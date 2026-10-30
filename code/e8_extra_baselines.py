"""e8_extra_baselines.py — 回应审稿意见 M4 / M6 的两个决定性对照

M4：金字塔（coarse-to-fine）相关能否解决大位移的分支问题？
    注意：对**严格周期**纹理，下采样同时缩小 p 与 Δ，Δ/p 不变，
    因此先验上判断歧义不会被尺度消除；但必须实测而非断言。
M6：一个零参数的几何弃权规则（按 |φ̂|/p 阈值）是否等效于学习式可观测性头？
    若等效，则"learned"没有增量。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

import e1_baselines as B
from e1_hybrid import build, features, coverage_curve
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"


def disp_band_profile(a: np.ndarray, b: np.ndarray, p: float) -> float:
    """给定 1D 剖面，返回 [−p/2, p/2) 内的互相关峰（带亚像素）。"""
    a = (a - a.mean()) * np.hanning(len(a))
    b = (b - b.mean()) * np.hanning(len(b))
    A, Bf = np.fft.rfft(a, n=2 * len(a)), np.fft.rfft(b, n=2 * len(b))
    c = np.fft.irfft(Bf * np.conj(A), n=2 * len(a))
    c = np.concatenate([c[-len(a) + 1:], c[:len(a)]])
    lags = np.arange(-len(a) + 1, len(a))
    m = np.abs(lags) <= max(1.0, p / 2)
    cc, ll = c[m], lags[m]
    i = int(np.argmax(cc))
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-12)
        return float(ll[i] + 0.5 * np.clip(d, -0.5, 0.5))
    return float(ll[i])


def disp_pyramid(I0: np.ndarray, I1: np.ndarray, p: float, levels: int = 3) -> float:
    """coarse-to-fine：粗层估位移，逐层上采样并在 ±1 px 内细化。"""
    a0, b0 = I0.mean(axis=0), I1.mean(axis=0)
    # 由粗到细
    est = 0.0
    for l in range(levels, 0, -1):
        f = 2.0 ** (l - 1)
        a = zoom(a0, 1.0 / f, order=1)
        b = zoom(b0, 1.0 / f, order=1)
        p_l = max(2.0, p / f)
        est_l = disp_band_profile(b, a, p_l)      # 注意 a/b 顺序：返回 a 相对 b 的位移
        est = (est + est_l) if l == levels else (est * 2.0 + est_l)
        # 下一层在父估计附近细化
        if l > 1:
            est *= 2.0
    # 末层用完整分辨率在 est ± 1.5 px 内细化
    a = (a0 - a0.mean()) * np.hanning(len(a0))
    b = (b0 - b0.mean()) * np.hanning(len(b0))
    lags = np.arange(-len(a) + 1, len(a))
    cand = np.abs(lags - est) <= 1.5
    A, Bf = np.fft.rfft(a, n=2 * len(a)), np.fft.rfft(b, n=2 * len(b))
    c = np.fft.irfft(Bf * np.conj(A), n=2 * len(a))
    c = np.concatenate([c[-len(a) + 1:], c[:len(a)]])
    cc, ll = c[cand], lags[cand]
    i = int(np.argmax(cc)) if len(cc) else 0
    return float(ll[i]) if len(ll) else est


def main() -> None:
    print("生成数据…")
    X, y, meta = build(n_per_cell=10)
    print(f"样本 {len(X)}")

    # ---------- M6：几何弃权规则 vs 学习式头
    phis = np.array([abs(m["phi"]) / m["pitch"] for m in meta])
    regimes = np.array([m["regime"] for m in meta])
    torch.manual_seed(20261010)
    # 用同一划分重训一个头，避免依赖旧 checkpoint 的结构
    from e7_head_ablation import MLPHead, train
    idx = np.random.default_rng(20261010).permutation(len(X))
    ntr = int(0.7 * len(X)); tr, te = idx[:ntr], idx[ntr:]
    torch.manual_seed(20261010)
    m = train(MLPHead(X.shape[1]), X[tr], y[tr])
    with torch.no_grad():
        s_learn = m(torch.from_numpy(X[te])).numpy()

    ratio_te = np.array([meta[i]["ratio"] for i in te])
    yte = y[te]
    print("\n=== M6：弃权策略对比（同一留出集）===")
    rows = {}
    for name, s in (("random", np.random.default_rng(0).random(len(te))),
                    ("|phi|/p rule", phis[te]),
                    ("|phi|/p + regime", phis[te] + 10.0 * (regimes[te] == "large")),
                    ("peak-ratio", -ratio_te),
                    ("learned head", s_learn)):
        r = coverage_curve(s, yte)
        rows[name] = r
        print(f"{name:20s} AURC {r['aurc']:.4f}  MAE@60% {r[0.6]:.4f}")

    # ---------- M4：金字塔相关 vs 单尺度
    print("\n=== M4：金字塔相关能否解决大位移（同 160 格点）===")
    rng = np.random.default_rng(7)
    agg = {"small": {"single": [[], []], "pyr": [[], []]},
           "large": {"single": [[], []], "pyr": [[], []]}}
    for p in B.PITCHES:
        for regime in ("small", "large"):
            lo, hi = (0.0, p / 2) if regime == "small" else (p / 2, 2 * p)
            for _ in range(10):
                d = float(rng.uniform(lo, hi))
                I0, I1 = B.render(p, d, rng)
                e1 = B.disp_band(I0, I1, p)
                e2 = disp_pyramid(I0, I1, p)
                kb = B.branch(d, p)
                for key, est in (("single", e1), ("pyr", e2)):
                    agg[regime][key][0].append(abs(est - d))
                    agg[regime][key][1].append(int(B.branch(est, p) != kb))
    print(f"{'regime':6s} {'method':8s} {'MAE':>7s} {'HAR':>6s}")
    for regime in ("small", "large"):
        for key in ("single", "pyr"):
            mae = float(np.mean(agg[regime][key][0]))
            har = float(np.mean(agg[regime][key][1]))
            print(f"{regime:6s} {key:8s} {mae:7.2f} {har:6.3f}")

    (OUT / "e8_extra.json").write_text(json.dumps(
        {"abstention": rows,
         "pyramid": {r: {k: {"mae": float(np.mean(v[0])), "har": float(np.mean(v[1]))}
                         for k, v in agg[r].items()} for r in agg}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e8_extra.json'}")


if __name__ == "__main__":
    main()
