"""e5_ablation.py — E5：可观测性头的特征消融（真实训练与评测）

消融对象（输入特征分组）：
  · 相关曲面 (64 维)
  · 退化统计 (高频能量、列均值离散度)
  · 峰值统计 (主峰、次峰、峰比)
  · 节距 p
指标：AURC（风险—覆盖率曲线下面积，越低越好）+ MAE@60% 覆盖率
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


def run(X, y, idx_tr, idx_te, cols):
    torch.manual_seed(SEED)
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
    print("生成数据…")
    X, y, meta = build()
    idx = np.random.default_rng(SEED).permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]

    all_cols = sum(GROUPS.values(), [])
    variants = {"Full model": all_cols}
    for name, cols in GROUPS.items():
        variants[f"w/o {name}"] = [c for c in all_cols if c not in cols]

    rows = []
    print(f"\n{'configuration':20s} {'AURC':>7s} {'MAE@60%':>8s}")
    for name, cols in variants.items():
        r = run(X, y, tr, te, cols)
        rows.append({"config": name, "aurc": r["aurc"], "mae60": r[0.6], "mae100": r[1.0]})
        print(f"{name:20s} {r['aurc']:7.4f} {r[0.6]:8.4f}")

    (OUT / "e5_ablation.json").write_text(
        json.dumps({"seed": SEED, "n": int(len(X)), "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n已写入 {OUT/'e5_ablation.json'}")


if __name__ == "__main__":
    main()
