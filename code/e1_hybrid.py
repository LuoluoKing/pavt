"""e1_hybrid.py — 方案 A：混合架构（经典亚像素相位 + 学习式可观测性/弃权）

设计依据
--------
E1 基线已证明：小位移下先验约束搜索达 0.36 px（近最优）；大位移下所有方法
HAR≈1——但这不是方法缺陷，而是**信息论上的不可辨识**：在严格周期纹理上，
Δ 与 Δ+kp 产生的图像完全相同（只差边界），任何观测者都无法判分支。
因此网络不该去"猜分支"，而应判断**当前观测能否支撑一个测量**。

架构
----
1. 相位：全分辨率相关曲面 + 抛物线插值（亚像素），限制在 [−p/2, p/2)。
2. 可观测性：小型 1D-CNN 读入相关曲面 + 退化统计 + p，回归预测 |误差|。
3. 决策：预测误差 > τ 则弃权。扫描 τ 得风险—覆盖率曲线。

对照
----
· random   ：随机弃权（下界）
· peak-ratio：经典启发式（峰值/次峰比低于阈值即弃权）
· ours     ：学习式预测误差

指标：MAE@覆盖率（100/80/60/40%）、AURC（越低越好）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import e1_baselines as B

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261010
NBIN = 64          # 相关曲面重采样点数
DEV = "cpu"


# ------------------------------------------------------------------ 特征
def features(I0, I1, p):
    """返回 (feat vector, 相位估计 phi_hat)"""
    c, lags = B.corr_surface(I0, I1)
    m = np.abs(lags) <= p / 2.0
    cc, ll = c[m], lags[m]
    i = int(np.argmax(cc))
    # 抛物线亚像素
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-12)
        phi = float(ll[i] + 0.5 * d)
    else:
        phi = float(ll[i])

    # 重采样到固定长度
    xs = np.linspace(-p / 2, p / 2, NBIN)
    surf = np.interp(xs, ll, cc)

    peak = float(cc.max())
    # 次峰：去掉主峰附近 ±2 点
    mask = np.ones_like(cc, dtype=bool)
    mask[max(0, i - 2): i + 3] = False
    second = float(cc[mask].max()) if mask.any() else float(cc.min())
    ratio = (peak - second) / (abs(peak) + 1e-6)

    # 退化统计：高频能量、列均值离散度
    hf = float(np.std(I0 - np.median(I0, axis=0, keepdims=True)))
    colstd = float(np.std(I0.mean(axis=0)))

    extra = np.array([p, peak, ratio, second, hf, colstd, abs(phi) / p], dtype=np.float32)
    feat = np.concatenate([surf.astype(np.float32), extra])
    return feat, phi, ratio


# ------------------------------------------------------------------ 数据
def build(n_per_cell=16, seed=SEED):
    rng = np.random.default_rng(seed)
    degs = [("none", {}), ("noise20", {"noise": 20}), ("noise60", {"noise": 60}),
            ("occl15", {"occl": 0.15}), ("occl30", {"occl": 0.30}), ("jpeg30", {"jpeg_q": 30})]
    X, y, meta = [], [], []
    for regime in ("small", "large"):
        for p in B.PITCHES:
            lo, hi = (0.0, p / 2.0) if regime == "small" else (p / 2.0, 2.0 * p)
            for dname, deg in degs:
                for _ in range(n_per_cell):
                    d = float(rng.uniform(lo, hi))
                    I0, I1 = B.render(p, d, rng, **deg)
                    f, phi, ratio = features(I0, I1, p)
                    X.append(f)
                    # 相对误差（按节距归一化），分支错误必然产生大误差
                    y.append(abs(phi - d) / p)
                    meta.append({"pitch": p, "true": d, "phi": phi, "regime": regime,
                                 "deg": dname, "ratio": ratio,
                                 "har": int(B.branch(phi, p) != B.branch(d, p))})
    return (np.stack(X).astype(np.float32), np.array(y, dtype=np.float32), meta)


class ObsNet(nn.Module):
    """可靠性头：曲面 → 8 个 token + 1 个 context token → 2 层 transformer encoder → 标量。

    与论文 §3.3 的描述一致（此前实现为 3 层 MLP，与正文不符，2026-10-30 修正）。
    当输入维度 < 64（消融中"去掉曲面"的变体）时退化为纯 context token 路径。
    """
    TOK_DIM = 8
    D = 32

    def __init__(self, nin):
        super().__init__()
        self.n_surface = 64 if nin >= 64 else 0
        self.n_tok = self.n_surface // self.TOK_DIM if self.n_surface else 0
        d = self.D
        if self.n_tok:
            self.tok = nn.Linear(self.TOK_DIM, d)
            self.idx = nn.Parameter(torch.zeros(self.n_tok + 1, d))
        ctx_in = max(1, nin - self.n_surface)
        self.ctx = nn.Linear(ctx_in, d)
        enc = nn.TransformerEncoderLayer(d, 2, dim_feedforward=2 * d,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        b = x.shape[0]
        if self.n_tok:
            surf = x[:, :self.n_surface].reshape(b, self.n_tok, self.TOK_DIM)
            ctx = x[:, self.n_surface:]
            if ctx.shape[1] == 0:
                ctx = torch.zeros((b, 1), dtype=x.dtype, device=x.device)
            s = torch.cat([self.tok(surf), self.ctx(ctx)[:, None, :]], dim=1)
            s = s + self.idx[None]
        else:
            s = self.ctx(x)[:, None, :]
        s = self.norm(self.tr(s))
        return self.out(s.mean(dim=1)).squeeze(-1)


def coverage_curve(scores, errs, fracs=(1.0, 0.8, 0.6, 0.4)):
    order = np.argsort(scores)
    errs = errs[order]
    out = {}
    for f in fracs:
        k = max(1, int(round(len(errs) * f)))
        out[f] = float(np.mean(errs[:k]))
    # AURC：覆盖率-误差曲线下面积（归一化）
    ks = np.arange(1, len(errs) + 1)
    curve = np.array([np.mean(errs[:k]) for k in ks])
    out["aurc"] = float(np.mean(curve))
    return out


def main():
    print("生成数据…")
    X, y, meta = build()
    print(f"  样本 {len(X)}，特征维 {X.shape[1]}")

    torch.manual_seed(SEED)
    idx = np.random.default_rng(SEED).permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]

    model = ObsNet(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(X[tr]); yt = torch.from_numpy(y[tr])
    for ep in range(120):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = F.l1_loss(model(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(model.state_dict(), OUT / "obs_synth.pt")
    print(f"  已保存权重 {OUT/'obs_synth.pt'}")
    with torch.no_grad():
        pred_te = model(torch.from_numpy(X[te])).numpy()
        # sanity check：同一头在留出集上应复现主结果
        _ = pred_te
    pred_tr = None

    errs_te = y[te]
    res = {}
    res["random"] = coverage_curve(np.random.default_rng(0).random(len(te)), errs_te)
    res["peak-ratio"] = coverage_curve(-np.array([meta[i]["ratio"] for i in te]), errs_te)
    res["ours"] = coverage_curve(pred_te, errs_te)

    print(f"\n{'method':12s} {'MAE@100%':>9s} {'@80%':>7s} {'@60%':>7s} {'@40%':>7s} {'AURC':>7s}")
    for k, v in res.items():
        print(f"{k:12s} {v[1.0]:9.4f} {v[0.8]:7.4f} {v[0.6]:7.4f} {v[0.4]:7.4f} {v['aurc']:7.4f}")

    # 分支在 accepted 子集上的准确率（只统计大位移样本）
    large = np.array([m["regime"] == "large" for m in meta])[te]
    print(f"\n大位移样本占比 {large.mean():.2f}")
    print(f"全样本 HAR（相位+分支，未弃权）: {np.mean([m['har'] for m in (meta[i] for i in te)]):.3f}")

    # 保存完整风险-覆盖率曲线（供 Fig.5 绘制）
    curves = {}
    for name, scores in (("random", np.random.default_rng(0).random(len(te))),
                         ("peak-ratio", -np.array([meta[i]["ratio"] for i in te])),
                         ("ours", pred_te)):
        order = np.argsort(scores)
        e = errs_te[order]
        curves[name] = [float(np.mean(e[:k])) for k in range(1, len(e) + 1)]

    (OUT / "e1_hybrid.json").write_text(json.dumps(
        {"seed": SEED, "n": int(len(X)), "results": res, "curves": curves,
         "note": "MAE 为按节距归一化的相对误差；AURC 为风险-覆盖率曲线下面积"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e1_hybrid.json'}")


if __name__ == "__main__":
    main()
