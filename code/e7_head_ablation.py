"""e7_head_ablation.py — 可靠性头的架构消融 + 多种子方差

自查发现两个必须补的实验：
1. **论文 §3.3 写的是两层 transformer encoder，代码里其实是 MLP** —— 必须让实现与描述一致，
   并把两种头放在同一数据上对比（这同时也是架构消融）。
2. 目前只有单一 70/30 划分，无方差 —— 改为 5 次随机划分，报告 mean ± sd。

指标：AURC（风险—覆盖率曲线下面积）与 MAE@60% 覆盖率。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e1_hybrid import build, coverage_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
SEED = 20261010
N_SPLIT = 5
D = 32
TOK_DIM = 8          # 64 维曲面 → 8 个 token


# ---------------------------------------------------------------- 两种头
class MLPHead(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, 128), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TransformerHead(nn.Module):
    """曲面 → tokens + 一个 context token → 2 层 transformer → 池化 → 标量。"""

    def __init__(self, nin, n_surface=64):
        super().__init__()
        self.n_tok = n_surface // TOK_DIM
        self.tok = nn.Linear(TOK_DIM, D)
        self.ctx = nn.Linear(nin - n_surface, D)
        enc = nn.TransformerEncoderLayer(D, 2, dim_feedforward=2 * D, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, 1)
        self.idx = nn.Parameter(torch.zeros(self.n_tok + 1, D))

    def forward(self, x):
        b = x.shape[0]
        surf = x[:, : self.n_tok * TOK_DIM].reshape(b, self.n_tok, TOK_DIM)
        ctx = x[:, self.n_tok * TOK_DIM:]
        s = torch.cat([self.tok(surf), self.ctx(ctx)[:, None, :]], dim=1) + self.idx[None]
        s = self.norm(self.tr(s))
        return self.out(s.mean(dim=1)).squeeze(-1)


def train(model, X, y, steps=600, bs=32, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (min(bs, len(xt)),))
        loss = F.l1_loss(model(xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main() -> None:
    print("生成数据…")
    X, y, meta = build()
    print(f"样本 {len(X)}")

    rng = np.random.default_rng(SEED)
    rows = []
    for s in range(N_SPLIT):
        idx = rng.permutation(len(X))
        ntr = int(0.7 * len(X))
        tr, te = idx[:ntr], idx[ntr:]
        ratio = np.array([meta[i]["ratio"] for i in te])
        yte = y[te]

        torch.manual_seed(SEED + s)
        mlp = train(MLPHead(X.shape[1]), X[tr], y[tr])
        torch.manual_seed(SEED + s)
        trh = train(TransformerHead(X.shape[1]), X[tr], y[tr])

        with torch.no_grad():
            p_mlp = mlp(torch.from_numpy(X[te])).numpy()
            p_trh = trh(torch.from_numpy(X[te])).numpy()

        for name, scores in (("random", np.random.default_rng(s).random(len(te))),
                             ("peak-ratio", -ratio),
                             ("mlp", p_mlp),
                             ("transformer", p_trh)):
            r = coverage_curve(scores, yte)
            rows.append({"split": s, "head": name, "aurc": r["aurc"], "mae60": r[0.6]})
        print(f"  split {s}: mlp AURC {rows[-3]['aurc']:.4f} | transformer AURC {rows[-1]['aurc']:.4f}")

    print(f"\n{'head':14s} {'AURC mean':>10s} {'± sd':>7s} {'MAE@60 mean':>12s}")
    summary = {}
    for head in ("random", "peak-ratio", "mlp", "transformer"):
        sel = [r for r in rows if r["head"] == head]
        a = np.array([r["aurc"] for r in sel]); m = np.array([r["mae60"] for r in sel])
        summary[head] = {"aurc_mean": float(a.mean()), "aurc_sd": float(a.std(ddof=1)),
                         "mae60_mean": float(m.mean())}
        print(f"{head:14s} {a.mean():10.4f} {a.std(ddof=1):7.4f} {m.mean():12.4f}")

    (OUT / "e7_head_ablation.json").write_text(json.dumps(
        {"n_split": N_SPLIT, "seed": SEED, "summary": summary, "rows": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e7_head_ablation.json'}")


if __name__ == "__main__":
    main()
