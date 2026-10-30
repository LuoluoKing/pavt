"""e2_real.py — E2：33 段真实扶梯视频上的零样本迁移

误差代理：**跨时间基线一致性**
   估计 Δ(t→t+G) 与 Δ(t→t+2G)；测量可靠时 |Δ(2G) − 2Δ(G)| 应很小。
   这是光流评测常用的 temporal consistency，且不会像前后向一致性那样
   在对称相关曲面上恒为零（已实测确认后者退化）。

两件事
------
1. 合成数据训练的可观测性头能否给真实视频的测量**排序**？→ Spearman + 风险—覆盖率
2. 与经典峰值—旁瓣比启发式对比

管线：2D 自相关选最周期 patch → 2D 带约束互相关 + 抛物线亚像素 →
按主方向旋转使运动水平 → 复用 e1_hybrid 的 71 维特征与网络打分。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import rotate
from scipy.stats import spearmanr

from e1_hybrid import ObsNet, features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "01_数据" / "raw"
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
OUT.mkdir(parents=True, exist_ok=True)

PATCH = 96
G = 4          # 基础时间基线（帧）
N_TRIPLE = 8
SEED = 20261010


def acf_score(img: np.ndarray) -> tuple[float, float]:
    x = img - img.mean()
    F = np.fft.rfft2(x, s=(2 * x.shape[0], 2 * x.shape[1]))
    acf = np.fft.irfft2(F * np.conj(F), s=(2 * x.shape[0], 2 * x.shape[1]))
    acf = acf[: x.shape[0], : x.shape[1]]
    acf /= acf[0, 0] + 1e-12
    h, w = acf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy, xx).astype(int)
    prof = np.bincount(r.ravel(), weights=acf.ravel()) / (np.bincount(r.ravel()) + 1e-9)
    for i in range(3, min(len(prof) - 2, 60)):
        if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > prof[i - 2] + 0.02:
            return float(prof[i]), float(i)
    return 0.0, 8.0


def disp2d(I0: np.ndarray, I1: np.ndarray, radius: float):
    a = I0 - I0.mean(); b = I1 - I1.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A = np.fft.rfft2(a * win); Bf = np.fft.rfft2(b * win)
    c = np.fft.fftshift(np.fft.irfft2(Bf * np.conj(A), s=a.shape))
    cy, cx = np.array(c.shape) // 2
    yy, xx = np.mgrid[0:c.shape[0], 0:c.shape[1]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= max(3.0, radius) ** 2
    cc = np.where(mask, c, -np.inf)
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    dy, dx = iy - cy, ix - cx
    peak = float(cc[iy, ix])

    def _par(v0, v1, v2):
        d = (v0 - v2) / (2 * (v0 - 2 * v1 + v2) + 1e-12)
        return float(np.clip(d, -0.5, 0.5)) if np.isfinite(d) else 0.0

    sy = _par(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]) if 0 < iy < c.shape[0] - 1 else 0.0
    sx = _par(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1]) if 0 < ix < c.shape[1] - 1 else 0.0

    tmp = cc.copy()
    tmp[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = -np.inf
    finite = tmp[np.isfinite(tmp)]
    second = float(finite.max()) if finite.size else float(c.min())
    return float(dx + sx), float(dy + sy), peak, second


def main() -> None:
    torch.manual_seed(SEED)
    model = ObsNet(71)
    ckpt = OUT / "obs_synth.pt"
    assert ckpt.exists(), f"缺少合成训练权重 {ckpt}，先跑 e1_hybrid.py"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    print(f"已加载合成训练权重：{ckpt.name}")
    videos = sorted(RAW.glob("*.mp4"))
    print(f"视频 {len(videos)} 段")
    rows = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n < 4 * G + 4:
            cap.release(); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2 - N_TRIPLE * G))
        frames = []
        for _ in range(N_TRIPLE * 2 * G + 4):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
        cap.release()
        if len(frames) < 2 * G + 2:
            continue

        best, best_p, best_xy = -1.0, 8.0, (0, 0)
        H, W = frames[0].shape
        for y in range(0, max(1, H - PATCH), 48):
            for x in range(0, max(1, W - PATCH), 48):
                s, p = acf_score(frames[0][y:y + PATCH, x:x + PATCH])
                if s > best:
                    best, best_p, best_xy = s, p, (x, y)
        x0, y0 = best_xy

        got = 0
        k = 0
        while k + 2 * G < len(frames) and got < N_TRIPLE:
            f0 = frames[k][y0:y0 + PATCH, x0:x0 + PATCH]
            f1 = frames[k + G][y0:y0 + PATCH, x0:x0 + PATCH]
            f2 = frames[k + 2 * G][y0:y0 + PATCH, x0:x0 + PATCH]
            r = max(3.0, best_p / 2)
            d1x, d1y, _, _ = disp2d(f0, f1, r)
            d2x, d2y, _, _ = disp2d(f0, f2, 2 * r)
            incons = float(np.hypot(d2x - 2 * d1x, d2y - 2 * d1y)) / (2 * G)

            ang = np.degrees(np.arctan2(d1y, d1x)) if (abs(d1x) + abs(d1y)) > 1e-6 else 0.0
            R0 = rotate(f0, -ang, reshape=False, order=1, mode="nearest")
            R1 = rotate(f1, -ang, reshape=False, order=1, mode="nearest")
            try:
                feat, _, ratio = features(R0, R1, best_p)
                with torch.no_grad():
                    score = float(model(torch.from_numpy(feat)[None])[0])
                fvec = [float(x) for x in feat]
            except Exception:
                score, ratio, fvec = float("nan"), float("nan"), None
            rows.append({"video": v.name, "acf": best, "pitch": best_p,
                         "d1": [d1x, d1y], "d2": [d2x, d2y],
                         "incons": incons, "score": score, "ratio": ratio,
                         "feat": fvec})
            got += 1
            k += G

    rows = [r for r in rows if np.isfinite(r["score"])]
    if not rows:
        print("无有效样本"); return
    incons = np.array([r["incons"] for r in rows])
    score = np.array([r["score"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    print(f"样本 {len(rows)}（{len({r['video'] for r in rows})} 段视频）| "
          f"不一致度 中位 {np.median(incons):.3f} | 均值 {incons.mean():.3f} | 90 分位 {np.quantile(incons,0.9):.3f}")

    if np.std(score) > 1e-9 and np.std(incons) > 1e-9:
        rho_s, p_s = spearmanr(score, incons)
        rho_r, p_r = spearmanr(-ratio, incons)
        print(f"Spearman(score, incons)   = {rho_s:+.3f} (p={p_s:.1e})")
        print(f"Spearman(-ratio, incons)  = {rho_r:+.3f} (p={p_r:.1e})")

    def curve(key, fracs=(1.0, 0.8, 0.6, 0.4)):
        idx = np.argsort(key)
        e = incons[idx]
        return {f: float(np.mean(e[:max(1, int(round(len(e) * f)))])) for f in fracs}

    res = {"random": curve(np.random.default_rng(0).random(len(rows))),
           "peak-ratio": curve(-ratio), "ours": curve(score)}
    print(f"\n{'policy':12s} {'@100%':>8s} {'@80%':>8s} {'@60%':>8s} {'@40%':>8s}")
    for k2, v2 in res.items():
        print(f"{k2:12s} {v2[1.0]:8.3f} {v2[0.8]:8.3f} {v2[0.6]:8.3f} {v2[0.4]:8.3f}")

    (OUT / "e2_real.json").write_text(json.dumps(
        {"n": len(rows), "n_video": len({r['video'] for r in rows}), "G": G,
         "curves": res, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'e2_real.json'}")


if __name__ == "__main__":
    main()
