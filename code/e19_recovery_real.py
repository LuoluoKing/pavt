"""e19_recovery_real.py — E3：真实扶梯纹理上的亚像素回收精度（回应审稿人 E3）

问题：真实数据无真值，"亚像素精度"与"校准后 51% 降低"都无法验证。
做法：取真实扶梯视频的 patch，用 FFT 循环平移施加**已知**位移，
      再用系统自己的相位阶段回收，测回收误差。这给了真实纹理上的真值。

协议
----
- 从 8 段真实视频各取 1 个高周期性 patch（96×96），共 8 个
- 已知位移 d ∈ {0.2, 0.4, 0.7, 1.0, 1.5, 2.0} px（沿 x）
- 用 e2_real.disp2d 在 ±p/2 内回收，测 |d̂ − d|
- 对照：整数像素最近邻（下界参考）与论文合成数据的 0.36 px
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from e2_real import acf_score, disp2d

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "01_数据" / "raw"
OUT = Path(__file__).resolve().parent / "results"
PATCH = 96
SHIFTS = [0.2, 0.4, 0.7, 1.0, 1.5, 2.0]


def shift_x(img, dx):
    n = img.shape[1]
    F = np.fft.rfft(img, axis=1)
    k = np.arange(F.shape[1])
    return np.fft.irfft(F * np.exp(-2j * np.pi * k * dx / n)[None, :], n, axis=1)


def main():
    videos = sorted(RAW.glob("*.mp4"))
    rows = []
    got = 0
    for v in videos:
        if got >= 8:
            break
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n < 30:
            cap.release(); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        H, W = g.shape
        if H < PATCH or W < PATCH:
            continue
        # 找最周期 patch
        best, best_p, best_xy = -1.0, 8.0, None
        for y in range(0, H - PATCH, 48):
            for x in range(0, W - PATCH, 48):
                s, p = acf_score(g[y:y + PATCH, x:x + PATCH])
                if s > best:
                    best, best_p, best_xy = s, p, (x, y)
        if best_xy is None or best < 0.14:
            continue
        x0, y0 = best_xy
        patch = g[y0:y0 + PATCH, x0:x0 + PATCH]
        radius = max(3.0, best_p / 2)

        errs = []
        for d in SHIFTS:
            moved = shift_x(patch, d)
            dh, _, _, _ = disp2d(patch, moved, radius)
            errs.append(abs(dh - d))
        rows.append({"video": v.name, "acf": float(best), "pitch": float(best_p),
                     "mae": float(np.mean(errs)), "max_err": float(np.max(errs)),
                     "per_shift": {str(s): float(e) for s, e in zip(SHIFTS, errs)}})
        got += 1
        print(f"{v.name}: p={best_p:.1f} acf={best:.2f} MAE={np.mean(errs):.3f} px", flush=True)

    if not rows:
        print("无可用 patch"); return
    maes = [r["mae"] for r in rows]
    print(f"\n{len(rows)} 个真实 patch | 回收 MAE 中位 {np.median(maes):.3f} px | "
          f"均值 {np.mean(maes):.3f} px | 最大 {np.max(maes):.3f} px")
    (OUT / "e19_recovery_real.json").write_text(json.dumps({
        "n_patch": len(rows), "shifts": SHIFTS,
        "median_mae": float(np.median(maes)), "mean_mae": float(np.mean(maes)),
        "max_mae": float(np.max(maes)), "rows": rows,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"已写入 {OUT/'e19_recovery_real.json'}")


if __name__ == "__main__":
    main()
