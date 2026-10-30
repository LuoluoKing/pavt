"""e17_ablation_5splits.py — 表 6 统一协议重跑：全部消融配置在 5 个随机划分下

统一 e5（单划分）与 e10（归一化对照）到同一协议：5 个 70/30 随机划分，
每个配置报告 AURC 与 MAE@60% 的均值 ± 标准差。配置：
  Full model / w/o surface / w/o pitch / w/o peak_stats / w/o degradation
  + surface normalized（按最大值归一化）+ peak height only
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from e1_hybrid import ObsNet, build, coverage_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"

GROUPS = {
    "surface": list(range(0, 64)),
    "pitch": [64],
    "peak_stats": [65, 66, 67],
    "degradation": [68, 69],
}
SEED = 20261010
N_SPLITS = 5


def run(X, y, idx_tr, idx_te, cols, seed):
    torch.manual_seed(seed)
    Xs = X[:, cols]
    m = ObsNet(Xs.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(Xs[idx_tr]), torch.from_numpy(y[idx_tr])
    for _ in range(120):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = F.l1_loss(m(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = m(torch.from_numpy(Xs[idx_te])).numpy()
    return coverage_curve(pred, y[idx_te])


def main():
    X, y, meta = build()
    Xn = X.copy()
    surf = Xn[:, :64]
    mx = np.maximum(surf.max(axis=1, keepdims=True), 1e-9)
    Xn[:, :64] = surf / mx
    peak_only = X[:, 65:66]  # 主峰高度

    all_cols = sum(GROUPS.values(), [])
    variants = {
        "Full model": (X, all_cols),
        "w/o surface": (X, [c for c in all_cols if c not in GROUPS["surface"]]),
        "w/o pitch": (X, [c for c in all_cols if c not in GROUPS["pitch"]]),
        "w/o peak_stats": (X, [c for c in all_cols if c not in GROUPS["peak_stats"]]),
        "w/o degradation": (X, [c for c in all_cols if c not in GROUPS["degradation"]]),
        "surface normalized": (Xn, all_cols),
        "peak height only": (peak_only, [0]),
    }

    rng = np.random.default_rng(SEED)
    splits = []
    for s in range(N_SPLITS):
        idx = rng.permutation(len(X))
        ntr = int(0.7 * len(X))
        splits.append((idx[:ntr], idx[ntr:], SEED + s))

    rows = []
    print(f"{'configuration':20s} {'AURC(mean±std)':>16s} {'MAE@60%(mean±std)':>20s}")
    for name, (Xv, cols) in variants.items():
        aurcs, mae60s = [], []
        for tr, te, sd in splits:
            r = run(Xv, y, tr, te, cols, sd)
            aurcs.append(r["aurc"])
            mae60s.append(r[0.6])
        rows.append({"config": name,
                     "aurc_mean": float(np.mean(aurcs)), "aurc_std": float(np.std(aurcs)),
                     "mae60_mean": float(np.mean(mae60s)), "mae60_std": float(np.std(mae60s)),
                     "aurc_all": [float(a) for a in aurcs]})
        print(f"{name:20s} {np.mean(aurcs):8.4f} ± {np.std(aurcs):.4f} "
              f"{np.mean(mae60s):10.4f} ± {np.std(mae60s):.4f}")

    (OUT / "e17_ablation_5splits.json").write_text(
        json.dumps({"seed": SEED, "n_splits": N_SPLITS, "n": int(len(X)), "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e17_ablation_5splits.json'}")


if __name__ == "__main__":
    main()
