"""e15c_simple_bound.py — 可实现上界：无需训练的边界残差判据（分离"证据存在性"与"学习效率"）

问题
----
e15 的 PAVT 正对照失败（positive 0.51），无法区分：
  (a) 任务不可学；(b) PAVT 架构不适合该任务。
需要一个**架构无关**的上界：如果证据存在，最简单的判据就应达到高准确率。

判据（两行代码的物理实现，无训练）
--------------------------------
对每个候选分支 k，假设位移 Δ_k = φ̂_next + k·p，
用该位移循环平移 I0 得到预测帧，比较预测与实测 I1 在**窗口边缘列**的残差：
    k̂ = argmin_k  mean|shift(I0, Δ_k) − I1|[:, edge]
边缘列是唯一携带非周期证据的地方（窗口外内容进入视野）。

预期（与 e15 的 diff 结构一一对应）
--------------------------------
  · negative, p|96 ：两假设逐位相同 → 两个 k 残差相同 → acc ≈ chance
  · negative, p∤96 ：结构性缝 → 正确 k 残差更小 → acc 高
  · positive       ：标记提供证据（所有 p）→ acc 高
若 positive 上界 ≈ 100% 而 PAVT 只到 0.51，则 PAVT 的失败是架构/训练问题，
不是"分支不可学"——这决定了论文负结果的表述边界。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"
W = H = 96
EDGE = 12  # 边缘列数


def periodic_profile_pure(width: int, p: float, rng) -> np.ndarray:
    x = np.arange(width, dtype=np.float64)
    duty = rng.uniform(0.35, 0.65)
    base = ((x / p) % 1.0 < duty).astype(np.float64) * 2.0 - 1.0
    a2, a3 = rng.uniform(0.1, 0.35), rng.uniform(0.05, 0.25)
    prof = base + a2 * np.sin(2 * np.pi * x / p + rng.uniform(0, 2 * np.pi)) \
                + a3 * np.sin(4 * np.pi * x / p + rng.uniform(0, 2 * np.pi))
    return prof / (np.abs(prof).max() + 1e-9)


def shift_x(img, dx):
    n = img.shape[1]
    F = np.fft.rfft(img, axis=1)
    k = np.arange(F.shape[1])
    return np.fft.irfft(F * np.exp(-2j * np.pi * k * dx / n)[None, :], n, axis=1)


def make_scene(p, rng, mode):
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


def render(scene, dx):
    img = shift_x(scene, dx)
    return (img - img.min()) / (img.max() - img.min() + 1e-12)


def boundary_classify(I0, I1, p, phi):
    """边界残差判据：在候选 k=0,1,2 中选边缘残差最小者。"""
    errs = []
    for k in range(3):
        pred = shift_x(I0, phi + k * p)
        e = np.abs(pred - I1)
        # 首尾各 EDGE 列
        err = float(np.concatenate([e[:, :EDGE], e[:, -EDGE:]], axis=1).mean())
        errs.append(err)
    return int(np.argmin(errs)), errs


def main():
    rng = np.random.default_rng(123)
    pitches = [8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 44, 50, 56, 60, 70, 80]
    rows = []
    print(f"{'mode':9s} {'p':>4s} {'96%p':>5s} {'boundary acc':>12s} {'n':>4s}")
    for mode in ("negative", "rowmod", "positive"):
        for p in pitches:
            acc, n = 0, 0
            for _ in range(40):
                d = float(rng.uniform(0, 2 * p))
                phi = d % p  # 相位（无歧义区间内可测）
                scene = make_scene(p, rng, mode)
                I0, I1 = render(scene, 0.0), render(scene, d)
                k_hat, _ = boundary_classify(I0, I1, p, phi)
                k_true = int(np.clip(np.floor(d / p), 0, 2))
                acc += int(k_hat == k_true)
                n += 1
            rows.append({"mode": mode, "pitch": p, "w_mod_p": W % p, "acc": acc / n, "n": n})
            print(f"{mode:9s} {p:4.0f} {W % p:5d} {acc/n:12.3f} {n:4d}", flush=True)

    summary = {}
    for mode in ("negative", "rowmod", "positive"):
        sel = [r for r in rows if r["mode"] == mode]
        div = [r["acc"] for r in sel if r["w_mod_p"] == 0]
        nod = [r["acc"] for r in sel if r["w_mod_p"] != 0]
        summary[mode] = {"mean_acc": float(np.mean([r["acc"] for r in sel])),
                         "mean_divisible": float(np.mean(div)),
                         "mean_nondivisible": float(np.mean(nod))}
        print(f"\n[{mode}] mean {summary[mode]['mean_acc']:.3f} | "
              f"divisible {summary[mode]['mean_divisible']:.3f} | "
              f"nondivisible {summary[mode]['mean_nondivisible']:.3f}")

    (OUT / "e15c_simple_bound.json").write_text(
        json.dumps({"edge_cols": EDGE, "rows": rows, "summary": summary},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e15c_simple_bound.json'}")


if __name__ == "__main__":
    main()
