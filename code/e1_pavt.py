"""e1_pavt.py — E1 阶段二：PAVT-lite 的训练与评测（CPU）

设计（与论文 3.2–3.4 节对应）
------------------------------
· 相位对齐位置编码：token 的 x 相位取 φ=(x mod p)/p，用傅里叶特征编码；
  平移 k·p 时 φ 不变 → 位置编码不变（论文的 PAT 内核）。
· 谐波对比：训练对 (I0, I1) 中 I1 相对 I0 平移 Δ∈[0,2p)，正样本即整周期平移；
  谐波头直接分类 k，相位头回归周期内相位 —— 二者相加得绝对位移。
· 可观测性头：预测 |Δ̂−Δ|/p，用于拒识。

与基线共用 e1_baselines 的数据生成器，保证同一分布、同一指标。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import e1_baselines as B

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
OUT.mkdir(parents=True, exist_ok=True)

DEV = "cpu"
H = W = 96
TOK = 4          # token 边长（8→4：保住亚像素相位信息）
GRID = W // TOK  # 12
D = 64
NHEAD = 4
NLAYER = 2
K_MAX = 3        # 谐波索引类别数（k = 0..K_MAX-1；Δ<2p 时 k∈{0,1,2}）
FOURIER_L = 4    # 相位傅里叶阶数


# ------------------------------------------------------------------ 数据
def sample_batch(bs: int, rng: np.random.Generator, degrade: bool = True):
    """返回 frames (B,2,1,H,W), pitch (B,), delta (B,)"""
    x0 = np.empty((bs, H, W), np.float32)
    x1 = np.empty((bs, H, W), np.float32)
    ps = np.empty(bs, np.float32)
    ds = np.empty(bs, np.float32)
    for i in range(bs):
        p = float(rng.choice(B.PITCHES))
        d = float(rng.uniform(0.0, 2.0 * p))
        kw = {}
        if degrade:
            r = rng.random()
            if r < 0.25:
                kw["noise"] = float(rng.choice([10, 20, 40, 60]))
            elif r < 0.4:
                kw["occl"] = float(rng.uniform(0.05, 0.25))
            elif r < 0.55:
                kw["jpeg_q"] = int(rng.choice([40, 30, 20]))
        I0, I1 = B.render(p, d, rng, **kw)
        x0[i], x1[i] = I0.astype(np.float32), I1.astype(np.float32)
        ps[i], ds[i] = p, d
    fr = torch.from_numpy(np.stack([x0, x1], axis=1))[:, :, None]   # (B,2,1,H,W)
    return fr, torch.from_numpy(ps), torch.from_numpy(ds)


def phase_features(p: torch.Tensor, device=DEV) -> torch.Tensor:
    """(B, GRID*GRID, 2*FOURIER_L) 周期锁定的 x 相位编码（y 只给行索引）。"""
    xs = (torch.arange(GRID) * TOK + TOK / 2.0).to(device)          # 列中心
    p = p.to(device)[:, None]                                        # (B,1)
    phi = torch.remainder(xs[None, :], p) / p                        # (B,GRID)
    feats = []
    for l in range(1, FOURIER_L + 1):
        feats += [torch.sin(2 * torch.pi * l * phi), torch.cos(2 * torch.pi * l * phi)]
    fx = torch.stack(feats, dim=-1)                                  # (B,GRID,2L)
    fx = fx[:, None, :, :].expand(-1, GRID, -1, -1)                  # (B,GRID,GRID,2L)
    return fx.reshape(p.shape[0], GRID * GRID, -1)


# ------------------------------------------------------------------ 模型
class PAVT(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, D, 3, stride=1, padding=1), nn.GELU(),   # /4 -> 24x24
        )
        self.phase_proj = nn.Linear(2 * FOURIER_L, D)
        self.row_emb = nn.Parameter(torch.zeros(GRID, D))
        enc = nn.TransformerEncoderLayer(D, NHEAD, dim_feedforward=2 * D,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, NLAYER)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Sequential(nn.Linear(6 * D + 2 * GRID - 1, 128), nn.GELU(), nn.Linear(128, 2 + K_MAX + 1))

    def forward(self, frames: torch.Tensor, p: torch.Tensor):
        b = frames.shape[0]
        x = frames.reshape(b * 2, 1, H, W)
        f = self.stem(x).flatten(2).transpose(1, 2)                  # (2B, N, D)
        pe = self.phase_proj(phase_features(p.repeat_interleave(2)))  # (2B,N,D)
        row = self.row_emb[:, None, :].repeat(1, GRID, 1).reshape(1, GRID * GRID, D)
        f = f + pe + row
        f = self.norm(self.tr(f))
        # --- 跨帧代价体积：沿水平 token 位移做匹配（这是判定谐波分支所必需的信息）
        F0, F1 = f[:b], f[b:]
        A0 = F0.reshape(b, GRID, GRID, D)
        A1 = F1.reshape(b, GRID, GRID, D)
        A0 = A0 / A0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        A1 = A1 / A1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vol = []
        for s in range(-(GRID - 1), GRID):
            if s >= 0:
                a, c = A0[:, :, s:, :], A1[:, :, :GRID - s, :]
            else:
                a, c = A0[:, :, :GRID + s, :], A1[:, :, -s:, :]
            vol.append((a * c).sum(-1).mean(dim=(1, 2)))
        vol = torch.stack(vol, dim=-1)                     # (b, 2*GRID-1)
        g = torch.cat([f.mean(dim=1), f.max(dim=1).values], dim=-1).reshape(b, 2, 2 * D)
        feat = torch.cat([g[:, 0], g[:, 1], (g[:, 1] - g[:, 0]).abs(), vol], dim=-1)
        o = self.head(feat)
        phase = o[:, :2]
        logits = o[:, 2:2 + K_MAX]
        obs = o[:, 2 + K_MAX]
        return phase, logits, obs


def decode(phase, logits, p):
    """(sin,cos) -> φ∈[0,1); 与谐波索引合成绝对位移。"""
    phi = torch.atan2(phase[:, 0], phase[:, 1]) / (2 * torch.pi)
    phi = torch.remainder(phi, 1.0)
    k = logits.argmax(-1).float()
    return phi * p + k * p, k


# ------------------------------------------------------------------ 训练
def train(steps: int = 320, bs: int = 16, lr: float = 3e-4, seed: int = 20261010):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PAVT().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    for it in range(1, steps + 1):
        fr, p, d = sample_batch(bs, rng)
        phase, logits, obs = model(fr, p)
        phi_true = torch.remainder(d, p) / p
        target = torch.stack([torch.sin(2 * torch.pi * phi_true),
                              torch.cos(2 * torch.pi * phi_true)], dim=-1)
        l_ph = F.mse_loss(phase, target)
        k_true = torch.clamp((d / p).floor(), 0, K_MAX - 1).long()
        l_k = F.cross_entropy(logits, k_true)
        with torch.no_grad():
            pred, _ = decode(phase, logits, p)
            err = (pred - d).abs() / p
        l_obs = F.l1_loss(obs, err)
        loss = l_ph + 0.5 * l_k + 0.3 * l_obs
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 150 == 0:
            print(f"  step {it:4d} loss {loss.item():.4f} | phase {l_ph.item():.4f} "
                  f"idx {l_k.item():.3f} obs {l_obs.item():.3f} | {time.time()-t0:.0f}s")
    return model


# ------------------------------------------------------------------ 评测
@torch.no_grad()
def evaluate(model, cells, n_per=12, seed: int = 7):
    rng = np.random.default_rng(seed)
    model.eval()
    rows = []
    for p, regime, deg in cells:
        lo, hi = (0.0, p / 2.0) if regime == "small" else (p / 2.0, 2.0 * p)
        errs, hits = [], 0
        obs_err, rep_err = [], []
        for _ in range(n_per):
            d = float(rng.uniform(lo, hi))
            I0, I1 = B.render(p, d, rng, **deg)
            fr = torch.from_numpy(np.stack([I0, I1])[None, :, None].astype(np.float32))
            pt = torch.tensor([p], dtype=torch.float32)
            phase, logits, obs = model(fr, pt)
            pred, k = decode(phase, logits, pt)
            e = float(pred[0] - d)
            errs.append(abs(e))
            hits += int(B.branch(float(pred[0]), p) != B.branch(d, p))
            q = float(obs[0])
            obs_err.append((q, abs(e)))
        ma = float(np.mean(errs))
        # 简单拒识：按 obs 排序，去掉最差 20% 后的 MAE
        pairs = sorted(obs_err)
        keep = pairs[: int(len(pairs) * 0.8)]
        rows.append({"pitch": p, "regime": regime, "degradation": deg,
                     "mae": ma, "har": hits / n_per,
                     "mae_at_80pct_coverage": float(np.mean([x[1] for x in keep]))})
    return rows


def main() -> None:
    print("训练 PAVT-lite（CPU）…")
    model = train()
    torch.save(model.state_dict(), OUT / "pavt_lite.pt")

    degs = [("none", {}), ("noise20", {"noise": 20}), ("occl15", {"occl": 0.15})]
    cells = []
    for regime in ("small", "large"):
        for p in (8.0, 12.0, 16.0, 24.0, 32.0, 44.0, 56.0, 70.0):
            for _, deg in degs:
                cells.append((p, regime, deg))
    rows = evaluate(model, cells)

    print(f"\n{'regime':6s} {'degradation':10s} | {'MAE':>7s} {'HAR':>6s} {'MAE@80%cov':>11s}")
    for regime in ("small", "large"):
        for dname, deg in degs:
            sel = [r for r in rows if r["regime"] == regime and r["degradation"] == deg]
            print(f"{regime:6s} {dname:10s} | {np.mean([r['mae'] for r in sel]):7.2f} "
                  f"{np.mean([r['har'] for r in sel]):6.3f} "
                  f"{np.mean([r['mae_at_80pct_coverage'] for r in sel]):11.2f}")

    (OUT / "e1_pavt.json").write_text(
        json.dumps({"seed": 20261010, "cells": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n已写入 {OUT/'e1_pavt.json'}")


if __name__ == "__main__":
    main()
