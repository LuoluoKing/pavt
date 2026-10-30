"""e9_decisive.py — 两个决定性对照

M9（机制）：把相关曲面按自身最大值归一化后重训，若 AURC 从 0.34 掉到随机，
          说明此前"读曲面形状"的说法不成立，模型只是在用峰高阈值。
M0′（动机）：在 G=1/4/8 三档时间基线下测 ‖Δ‖/p 的分布，回答"歧义是不是
          由基线选择制造的"，以及"帧间工况是否根本不需要弃权"。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import rotate
from scipy.stats import spearmanr

from e1_hybrid import ObsNet, build, coverage_curve, features
from e7_head_ablation import train as train_head

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "01_数据" / "raw"
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
SEED = 20261010
PATCH = 96


# ---------------------------------------------------------------- M9
def m9_normalised_surface():
    print("=== M9：曲面自归一化对照 ===")
    X, y, meta = build()
    Xn = X.copy()
    for i in range(len(Xn)):
        s = Xn[i, :64]
        m = np.abs(s).max()
        if m > 1e-6:
            Xn[i, :64] = s / m           # 只归一化曲面，其余 7 个标量不动

    idx = np.random.default_rng(SEED).permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    res = {}
    for name, Xin in (("raw surface", X), ("max-normalised surface", Xn)):
        torch.manual_seed(SEED)
        m = train_head(ObsNet(Xin.shape[1]), Xin[tr], y[tr])
        with torch.no_grad():
            s = m(torch.from_numpy(Xin[te])).numpy()
        r = coverage_curve(s, y[te])
        res[name] = r
        print(f"  {name:24s} AURC {r['aurc']:.4f}  MAE@60% {r[0.6]:.4f}")

    # 单一特征对照：纯峰高
    peak = X[:, 65]
    r = coverage_curve(-peak, y)
    res["peak height only"] = r
    print(f"  {'peak height only':24s} AURC {r['aurc']:.4f}  MAE@60% {r[0.6]:.4f}")
    return res


# ---------------------------------------------------------------- M0′
def m0_baselines():
    print("\n=== M0′：三档时间基线下 ‖Δ‖/p ===")
    rng = np.random.default_rng(SEED)
    out = {}
    for G in (1, 4, 8):
        vals, pitches = [], []
        for v in sorted(RAW.glob("*.mp4"))[:33]:
            cap = cv2.VideoCapture(str(v))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if n < 4 * G + 4:
                cap.release(); continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2 - 4 * G))
            fr = []
            for _ in range(4 * G + 4):
                ok, f = cap.read()
                if not ok:
                    break
                fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
            cap.release()
            if len(fr) < 2 * G + 2:
                continue
            # 取画面中心块（不作为节距选择，只为统一比较三档基线）
            H, W = fr[0].shape
            x0, y0 = max(0, W // 2 - PATCH // 2), max(0, H // 2 - PATCH // 2)
            f0 = fr[0][y0:y0 + PATCH, x0:x0 + PATCH]
            f1 = fr[G][y0:y0 + PATCH, x0:x0 + PATCH]
            feat, phi, ratio = features(f0, f1, 8.0)
            p_est = float(feat[64])
            # 用 2D 相关取位移
            from e2_real import disp2d
            d = disp2d(f0, f1, max(3.0, p_est))
            mag = float(np.hypot(d[0], d[1]))
            if p_est > 0:
                vals.append(mag / p_est)
                pitches.append(p_est)
        if vals:
            v = np.array(vals)
            out[f"G={G}"] = {"n": len(v), "median": float(np.median(v)),
                             "frac_gt_0.5": float((v > 0.5).mean()),
                             "pitch_median": float(np.median(pitches))}
            print(f"  G={G}: n={len(v)}  median |Δ|/p={np.median(v):.3f}  "
                  f">0.5={100*(v>0.5).mean():.1f}%  pitch中位={np.median(pitches):.1f}px")
    return out


if __name__ == "__main__":
    r9 = m9_normalised_surface()
    r0 = m0_baselines()
    (OUT / "e9_decisive.json").write_text(
        json.dumps({"m9": r9, "m0": r0}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e9_decisive.json'}")
