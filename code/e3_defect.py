"""e3_defect.py — E3：真实扶梯图像上的结构性缺陷检测（受控注入，有真值）

为什么不用公开数据集
------------------
论文 §3.5 的异常是"刚性周期结构的结构缺陷"（缺单元、异物、相位断裂）。
公开视频异常数据集（Avenue/Ped2）标注的是行人场景异常，与这一机制不同；
在自有真实图像上**受控注入**缺陷，可以获得精确真值与定位标注。

打分
----
把 96×96 结构块切成 32×32 子块（步长 16），对每个子块算：
  · 局部周期性强度 s = 径向 ACF 的首个显著局部极大值
  · 相位一致性：子块与整块参考列在 [−p/2, p/2) 内最佳对齐的相位残差
异常分数 = 1 − s（周期性被破坏），并记录最大响应的位置用于定位评估。

三类注入
--------
1. missing：用一个周期宽度的平滑块覆盖（模拟梯级缺失）
2. foreign：粘贴一块随机纹理（异物）
3. shift  ：把子区域沿运动方向平移 p/2（相位断裂）
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "01_数据" / "raw"
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
OUT.mkdir(parents=True, exist_ok=True)

PATCH, SUB, STRIDE = 96, 32, 16
N_FRAMES = 6
SEED = 20261010


def acf_strength(img: np.ndarray) -> tuple[float, float]:
    x = img - img.mean()
    F = np.fft.rfft2(x, s=(2 * x.shape[0], 2 * x.shape[1]))
    acf = np.fft.irfft2(F * np.conj(F), s=(2 * x.shape[0], 2 * x.shape[1]))
    acf = acf[: x.shape[0], : x.shape[1]]
    acf /= acf[0, 0] + 1e-12
    h, w = acf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy, xx).astype(int)
    prof = np.bincount(r.ravel(), weights=acf.ravel()) / (np.bincount(r.ravel()) + 1e-9)
    for i in range(3, min(len(prof) - 2, 40)):
        if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > prof[i - 2] + 0.02:
            return float(prof[i]), float(i)
    return 0.0, 8.0


def inject(patch: np.ndarray, kind: str, p: float, rng) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = patch.copy()
    h, w = patch.shape
    bw = max(8, int(round(p)))
    bh = max(16, int(round(p * 1.5)))
    x0 = int(rng.integers(4, w - bw - 4))
    y0 = int(rng.integers(4, h - bh - 4))
    if kind == "missing":
        # 用局部中值填充一个"缺单元"带
        fill = np.median(patch[max(0, y0 - 8):y0 + bh + 8, max(0, x0 - 8):x0 + bw + 8])
        out[y0:y0 + bh, x0:x0 + bw] = fill
    elif kind == "foreign":
        tex = rng.normal(0.5, 0.18, (bh, bw)).astype(np.float32)
        out[y0:y0 + bh, x0:x0 + bw] = np.clip(tex, 0, 1)
    else:  # shift：子区域沿 x 平移 p/2
        half = max(4, int(round(p / 2)))
        reg = patch[y0:y0 + bh, x0:x0 + bw]
        out[y0:y0 + bh, x0:x0 + bw] = np.roll(reg, half, axis=1)
    return out, (x0, y0, bw, bh)


def score_map(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (分数图, 坐标网格)，分数 = 1 − 局部周期性强度。"""
    h, w = patch.shape
    ys = list(range(0, h - SUB + 1, STRIDE))
    xs = list(range(0, w - SUB + 1, STRIDE))
    S = np.zeros((len(ys), len(xs)), np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            s, _ = acf_strength(patch[y:y + SUB, x:x + SUB])
            S[i, j] = 1.0 - s
    return S, np.array([xs, ys])


def auc(scores_pos, scores_neg) -> float:
    s = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s)
    y = y[order]
    # 秩和 AUC
    ranks = np.arange(1, len(y) + 1)
    pos_ranks = ranks[y == 1].sum()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((pos_ranks - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    rng = np.random.default_rng(SEED)
    videos = sorted(RAW.glob("*.mp4"))
    per_kind = {"missing": [[], []], "foreign": [[], []], "shift": [[], []]}
    local_hits = {"missing": [], "foreign": [], "shift": []}
    n_img = 0

    for v in videos[:20]:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n < 10:
            cap.release(); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, fr = cap.read(); cap.release()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        H, W = g.shape
        best, bp, bxy = -1, 8.0, (0, 0)
        for y in range(0, max(1, H - PATCH), 48):
            for x in range(0, max(1, W - PATCH), 48):
                s, p = acf_strength(g[y:y + PATCH, x:x + PATCH])
                if s > best:
                    best, bp, bxy = s, p, (x, y)
        if best < 0.15:
            continue
        patch = g[bxy[1]:bxy[1] + PATCH, bxy[0]:bxy[0] + PATCH]
        S0, grid = score_map(patch)
        n_img += 1
        for kind in per_kind:
            inj, box = inject(patch, kind, bp, rng)
            S1, _ = score_map(inj)
            xs, ys = grid
            x0, y0, bw, bh = box
            # 缺陷覆盖的子块索引
            hit = np.zeros_like(S1, dtype=bool)
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    if not (x + SUB <= x0 or x >= x0 + bw or y + SUB <= y0 or y >= y0 + bh):
                        hit[i, j] = True
            if hit.sum() == 0 or (~hit).sum() == 0:
                continue
            per_kind[kind][0].extend(S1[hit].ravel().tolist())
            per_kind[kind][1].extend(S1[~hit].ravel().tolist())
            # 定位：最大响应是否落在缺陷块内
            iy, ix = np.unravel_index(np.argmax(S1), S1.shape)
            local_hits[kind].append(int(hit[iy, ix]))

    rows = []
    print(f"图像 {n_img} 张\n")
    print(f"{'injection':10s} {'AUC':>6s} {'localisation':>13s}")
    for kind, (pos, neg) in per_kind.items():
        if not pos or not neg:
            continue
        a = auc(np.array(pos), np.array(neg))
        lh = float(np.mean(local_hits[kind])) if local_hits[kind] else float("nan")
        rows.append({"kind": kind, "auc": a, "loc": lh,
                     "n_pos": len(pos), "n_neg": len(neg)})
        print(f"{kind:10s} {a:6.3f} {lh:13.2f}")

    (OUT / "e3_defect.json").write_text(json.dumps(
        {"seed": SEED, "n_image": n_img, "sub": SUB, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n已写入 {OUT/'e3_defect.json'}")


if __name__ == "__main__":
    main()
