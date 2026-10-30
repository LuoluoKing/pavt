"""e15_fft_regen.py — P0 v2：FFT 精确循环平移重生成 + 正/负对照（回应审稿 P0-1）

背景
----
审稿人 P0-1 指出：旧生成器 `nd_shift(order=3, mode='wrap')` 的样条边界伪影
（e10 实测 19–22× 噪声）污染了可辨识性论证。本版用数学精确的 FFT 循环
平移重生成，并给出**正确设计**的正对照。

v2 修正（相对 v1）
------------------
- BUG-EXP15-01：验证段两次调用 render 各自消耗 rng → 对比了不同剖面。
  v2 把"场景构造"（消耗随机数）与"渲染"（纯确定）分离。
- BUG-EXP15-02：v1 的 positive 标记是**静止**的（两帧同一位置），不提供
  分支证据。v2 的标记属于**场景**（贴在结构上），随场景一起循环平移。

正确的可辨识性刻画（离散窗口 N=96）
------------------------------------
观测是 N 点离散信号；平移 = 频域相位斜坡 exp(-2πik·dx/N)。
两假设 Δ 与 Δ+p 相同 ⇔ 有能量频率都满足 kp ≡ 0 (mod N)。
  - p|96：对所有整数 k 成立 → 两假设逐位相同（算术必然，与内容无关）
  - p∤96：带限循环平移在窗口边界产生结构性缝 → 两假设相差窗口边界内容
三个条件：
  A. negative：场景 = 纯周期剖面（无标记）
     → p|96 两假设逐位相同；p∤96 相差结构性缝（e12 已验证 1.45–2.02 px 量级）
  B. positive：场景 = 剖面 + 附着标记（随场景平移）
     → 所有 p 两假设都相差标记内容 → 管线若有效应能学会（>90% 判据）
  C. rowmod：negative + 静止行调制（相机特性，两帧相同，不提供证据）
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import e1_baselines as B
from e1_pavt import K_MAX, PAVT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
W = H = 96
STEPS = 960
BS = 16
SEED = 20261010
MODES = ("rowmod", "positive")
# negative 已于 2026-10-29 完成（960 步），结果抢救在 e15_partial_negative.json，
# 本脚本结束时自动合并，不重复训练。


def periodic_profile_pure(width: int, p: float, rng) -> np.ndarray:
    """纯周期剖面（严格 p-周期采样）。"""
    x = np.arange(width, dtype=np.float64)
    duty = rng.uniform(0.35, 0.65)
    base = ((x / p) % 1.0 < duty).astype(np.float64) * 2.0 - 1.0
    a2, a3 = rng.uniform(0.1, 0.35), rng.uniform(0.05, 0.25)
    prof = base + a2 * np.sin(2 * np.pi * x / p + rng.uniform(0, 2 * np.pi)) \
                + a3 * np.sin(4 * np.pi * x / p + rng.uniform(0, 2 * np.pi))
    return prof / (np.abs(prof).max() + 1e-9)


def shift_x(img: np.ndarray, dx: float) -> np.ndarray:
    """2D 图像沿 x 的 FFT 精确循环平移（逐行 1D）。"""
    n = img.shape[1]
    F = np.fft.rfft(img, axis=1)
    k = np.arange(F.shape[1])
    F = F * np.exp(-2j * np.pi * k * dx / n)[None, :]
    return np.fft.irfft(F, n, axis=1)


def make_scene(p: float, rng, mode: str) -> np.ndarray:
    """构造场景（消耗随机数，只调用一次）。场景 = 剖面 + 附着标记。"""
    prof = periodic_profile_pure(W, p, rng)
    scene = np.repeat(prof[None, :], H, axis=0)
    if mode == "positive":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        for _ in range(3):
            cy, cx = rng.uniform(10, H - 10), rng.uniform(10, W - 10)
            s = rng.uniform(3, 6)
            scene += 0.5 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    if mode == "rowmod":
        scene = scene + 0.08 * np.sin(2 * np.pi * np.arange(H)[:, None] / rng.uniform(30, 60))
    return scene


def render(scene: np.ndarray, dx: float, mode: str) -> np.ndarray:
    """确定性渲染：场景循环平移 + （rowmod 的静止调制已含在场景中）。"""
    img = shift_x(scene, dx)
    return ((img - img.min()) / (img.max() - img.min() + 1e-12)).astype(np.float32)


def sample_batch(bs, rng, mode):
    x0 = np.empty((bs, H, W), np.float32)
    x1 = np.empty((bs, H, W), np.float32)
    ps = np.empty(bs, np.float32)
    ds = np.empty(bs, np.float32)
    for i in range(bs):
        p = float(rng.choice(B.PITCHES))
        d = float(rng.uniform(0.0, 2.0 * p))
        scene = make_scene(p, rng, mode)
        x0[i], x1[i] = render(scene, 0.0, mode), render(scene, d, mode)
        ps[i], ds[i] = p, d
    fr = torch.from_numpy(np.stack([x0, x1], axis=1)[:, None])
    return fr, torch.from_numpy(ps), torch.from_numpy(ds)


def train(mode: str, steps: int = STEPS, seed: int = SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PAVT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    for it in range(1, steps + 1):
        fr, p, d = sample_batch(BS, rng, mode)
        _, logits, _ = model(fr, p)
        k_true = torch.clamp((d / p).floor(), 0, K_MAX - 1).long()
        loss = F.cross_entropy(logits, k_true)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 200 == 0:
            print(f"  [{mode}] step {it:4d} ce {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def eval_branch(model, mode, n_per=24, seed=123):
    rng = np.random.default_rng(seed)
    model.eval()
    rows = []
    for p in B.PITCHES:
        acc, n = 0, 0
        for _ in range(n_per):
            d = float(rng.uniform(0, 2 * p))
            scene = make_scene(p, rng, mode)
            I0, I1 = render(scene, 0.0, mode), render(scene, d, mode)
            fr = torch.from_numpy(np.stack([I0, I1])[None, :, None].astype(np.float32))
            pt = torch.tensor([p], dtype=torch.float32)
            _, logits, _ = model(fr, pt)
            k_hat = int(logits.argmax(-1))
            k_true = int(np.clip(np.floor(d / p), 0, K_MAX - 1))
            acc += int(k_hat == k_true)
            n += 1
        rows.append({"pitch": p, "w_mod_p": W % p, "acc": acc / n})
        print(f"  [{mode}] p={p:4.0f} acc={acc/n:.3f}", flush=True)
    return rows


def main():
    # ---------- 0. 数值验证（同一场景，修复版） ----------
    print("=== 数值验证 max|I(dx) - I(dx+p)|（同一场景，BUG-EXP15-01 已修） ===", flush=True)
    rng = np.random.default_rng(0)
    verify = []
    for mode in MODES:
        for p in (8.0, 16.0, 33.0, 70.0):
            diffs = []
            for _ in range(10):
                d = float(rng.uniform(0, 2 * p))
                scene = make_scene(p, rng, mode)
                Ia, Ib = render(scene, d, mode), render(scene, d + p, mode)
                diffs.append(float(np.abs(Ia - Ib).max()))
            verify.append({"mode": mode, "pitch": p, "max_diff": float(np.mean(diffs))})
            print(f"  {mode:9s} p={p:4.0f}  mean max|diff| = {np.mean(diffs):.6f}", flush=True)

    # ---------- 训练 + 评测 ----------
    results = {}
    for mode in MODES:
        print(f"\n=== 训练 {mode}（{STEPS} 步）===", flush=True)
        model = train(mode)
        rows = eval_branch(model, mode)
        accs = [r["acc"] for r in rows]
        div = [r["acc"] for r in rows if r["w_mod_p"] == 0]
        nodiv = [r["acc"] for r in rows if r["w_mod_p"] != 0]
        results[mode] = {
            "mean_acc": float(np.mean(accs)),
            "mean_acc_divisible": float(np.mean(div)) if div else None,
            "mean_acc_nondivisible": float(np.mean(nodiv)) if nodiv else None,
            "per_pitch": rows,
        }
        print(f"  [{mode}] 平均 acc = {np.mean(accs):.3f} "
              f"(整除组 {np.mean(div) if div else float('nan'):.3f} / 非整除组 {np.mean(nodiv) if nodiv else float('nan'):.3f})", flush=True)

    out = dict(seed=SEED, steps=STEPS, k_max=K_MAX, note="v2: BUG-EXP15-01/02 fixed",
               verify=verify, results=results)
    # 合并 2026-10-29 已完成的 negative 结果
    partial = OUT / "e15_partial_negative.json"
    if partial.exists():
        neg = json.loads(partial.read_text(encoding="utf-8"))["negative"]
        out["results"]["negative"] = neg
        print("已合并 negative（来自 e15_partial_negative.json）", flush=True)
    (OUT / "e15_fft_regen.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e15_fft_regen.json'}")


if __name__ == "__main__":
    main()
