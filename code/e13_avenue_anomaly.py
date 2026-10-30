"""e13_avenue_anomaly.py — E13: CUHK Avenue 上的帧级异常检测（§3.5/§5.4 恢复实验）

协议（Avenue 标准）
------------------
- 训练：16 个 training_vol（全部正常）
- 测试：21 个 testing_vol，帧级 GT 来自 ground_truth_demo/testing_label_mask/{i}_label.mat
  （volLabel 为 1×T cell，帧级标签 = 该帧掩码是否非零）
- 指标：全测试集合并帧级 ROC AUC（主指标）+ 逐视频 AUC（中位数/最小）

方法（同一打分框架，三变体公平对比）
----------------------------------
统一流程：32×32 patch、步长 16（网格 6×9=54 patches/帧）→ 描述符 → 与
memory bank 的 kNN 距离 → 帧分数 = top-q patch 均值（q=10%）。

1. ours_periodic : 描述符 = [2×ACF 径向剖面(16 bins), 1×外观(8×8 均值池化, 64 维)]
   周期感知描述符——论文框架中"相位场/周期结构"通道的 memory-bank 实例化。
2. base_appear   : 描述符 = 外观 only（无 ACF）。分离周期通道的增益。
3. base_framediff: 相邻帧差能量（高斯平滑后 Σ|I_t − I_{t-1}|）。经典无学习基线。

诚实性约束
----------
- Avenue 是行人场景（非刚性周期结构）。实验检验的是"同一打分框架在一般
  真实场景的行为"，不声称是扶梯周期监测。
- 所有数字直接来自本脚本真实运行，写入 results/e13_avenue_anomaly.json。
- 无 GPU、单机 CPU。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "01new_frai" / "data" / "Avenue Dataset"
GT_DIR = ROOT / "01new_frai" / "data" / "ground_truth_demo" / "testing_label_mask"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

SEED = 20261010
PATCH = 32
STEP = 16
ACF_BINS = 16
APP_DIM = 8  # 8x8 池化 -> 64 维
TOP_Q = 0.10  # 帧分数 = top 10% patch 均值
N_TRAIN_FRAMES_PER_VOL = 150
ACF_W = 2.0  # ACF 通道权重

GRID_R = (120 - PATCH) // STEP + 1  # 6
GRID_C = (160 - PATCH) // STEP + 1  # 9
N_PATCH = GRID_R * GRID_C


def load_gt(video_id: int) -> np.ndarray:
    """帧级标签 (T,)，1=异常帧。"""
    g = sio.loadmat(GT_DIR / f"{video_id}_label.mat")["volLabel"]
    T = g.shape[1]
    lab = np.zeros(T, dtype=np.int8)
    for t in range(T):
        m = np.asarray(g[0, t])
        lab[t] = 1 if (m > 0).any() else 0
    return lab


def frame_grid(vol: np.ndarray, frames: np.ndarray):
    """取若干帧并切 patch 网格。返回 (n_frames, n_patch, 32, 32) float32。"""
    fr = vol[:, :, frames].astype(np.float32) / 255.0  # (120,160,n)
    fr = fr.transpose(2, 0, 1)  # (n,120,160)
    patches = np.empty((len(frames), N_PATCH, PATCH, PATCH), dtype=np.float32)
    k = 0
    for r in range(GRID_R):
        for c in range(GRID_C):
            patches[:, k] = fr[:, r * STEP:r * STEP + PATCH, c * STEP:c * STEP + PATCH]
            k += 1
    return patches


def acf_profile(patches: np.ndarray) -> np.ndarray:
    """径向 ACF 剖面 (n, n_patch, ACF_BINS)。patches: (n, n_patch, 32, 32)"""
    n, np_, P, _ = patches.shape
    x = patches - patches.mean(axis=(2, 3), keepdims=True)
    F = np.fft.rfft2(x, s=(2 * P, 2 * P), axes=(2, 3))
    ac = np.fft.irfft2(F * np.conj(F), s=(2 * P, 2 * P), axes=(2, 3))
    ac = np.fft.fftshift(ac, axes=(2, 3))  # 中心在 (P, P)
    c = P
    yy, xx = np.mgrid[0:2 * P, 0:2 * P]
    rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    prof = np.empty((n, np_, ACF_BINS), dtype=np.float32)
    for b in range(ACF_BINS):
        m = (rr >= b + 0.5) & (rr < b + 1.5)
        prof[:, :, b] = ac[:, :, m].mean(axis=2)
    # 归一化：r=0 处为自相关峰值
    prof = prof / np.maximum(prof[:, :, 0:1], 1e-8)
    return prof


def appearance(patches: np.ndarray) -> np.ndarray:
    """8×8 均值池化外观 (n, n_patch, 64)。"""
    n, np_, P, _ = patches.shape
    b = P // APP_DIM  # 4
    ap = patches.reshape(n, np_, APP_DIM, b, APP_DIM, b).mean(axis=(3, 5))
    return ap.reshape(n, np_, APP_DIM * APP_DIM)


def descriptors(patches: np.ndarray, with_acf: bool) -> np.ndarray:
    """拼接描述符并做通道加权。返回 (n*n_patch, D)。"""
    ap = appearance(patches)  # (n, n_patch, 64)
    if with_acf:
        pf = acf_profile(patches)  # (n, n_patch, 16)
        d = np.concatenate([pf * ACF_W, ap], axis=2)
    else:
        d = ap
    n = d.shape[0]
    return d.reshape(n * N_PATCH, -1)


def frame_scores_from_desc(desc: np.ndarray, tree: cKDTree) -> np.ndarray:
    """(n_frames*n_patch, D) -> (n_frames,) 帧分数 = top-q patch kNN 距离均值。"""
    dist, _ = tree.query(desc, k=1, workers=-1)
    ds = dist.reshape(-1, N_PATCH)
    q = max(1, int(round(TOP_Q * N_PATCH)))
    top = np.sort(ds, axis=1)[:, -q:]  # 最大的 q 个
    return top.mean(axis=1)


def framediff_scores(vol: np.ndarray) -> np.ndarray:
    """相邻帧差能量 (T,)。首帧无前帧，复制第二帧的差值以保持长度对齐。"""
    x = vol.astype(np.float32) / 255.0  # (120,160,T)
    d = np.abs(np.diff(x, axis=2)).sum(axis=(0, 1))  # (T-1,)
    return np.concatenate([[d[0]], d])  # (T,)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC（Mann-Whitney，带 tie 处理）。"""
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # ties 平均秩
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ---------- 训练（memory bank）----------
    train_ids = list(range(1, 17))
    bank_p, bank_a = [], []
    for vid in train_ids:
        vol = sio.loadmat(DATA / "training_vol" / f"vol{vid:02d}.mat")["vol"]
        T = vol.shape[2]
        frames = np.linspace(0, T - 1, N_TRAIN_FRAMES_PER_VOL).astype(int)
        patches = frame_grid(vol, frames)
        bank_p.append(descriptors(patches, with_acf=True))
        bank_a.append(descriptors(patches, with_acf=False))
        print(f"[train vol{vid:02d}] T={T} frames={len(frames)} bank_p={bank_p[-1].shape}", flush=True)
    bank_p = np.concatenate(bank_p, axis=0)
    bank_a = np.concatenate(bank_a, axis=0)
    print(f"bank_p: {bank_p.shape}, bank_a: {bank_a.shape}, {time.time()-t0:.0f}s", flush=True)
    tree_p = cKDTree(bank_p)
    tree_a = cKDTree(bank_a)

    # ---------- 测试 ----------
    test_ids = list(range(1, 22))
    all_y, all_s_p, all_s_a, all_s_d, per_video = [], [], [], [], []
    for vid in test_ids:
        vol = sio.loadmat(DATA / "testing_vol" / f"vol{vid:02d}.mat")["vol"]
        y = load_gt(vid)
        T = vol.shape[2]
        assert T == len(y), f"vol{vid:02d}: vol T={T} vs GT T={len(y)}"
        frames = np.arange(T)
        patches = frame_grid(vol, frames)
        s_p = frame_scores_from_desc(descriptors(patches, True), tree_p)
        s_a = frame_scores_from_desc(descriptors(patches, False), tree_a)
        s_d = framediff_scores(vol)
        auc_p = roc_auc(y, s_p)
        auc_a = roc_auc(y, s_a)
        auc_d = roc_auc(y, s_d)
        per_video.append(dict(video=vid, n_frames=T, n_abn=int(y.sum()),
                              auc_ours=auc_p, auc_appear=auc_a, auc_framediff=auc_d))
        all_y.append(y)
        all_s_p.append(s_p)
        all_s_a.append(s_a)
        all_s_d.append(s_d)
        print(f"[test vol{vid:02d}] T={T} abn={int(y.sum())} AUC ours={auc_p:.3f} appear={auc_a:.3f} framediff={auc_d:.3f} ({time.time()-t0:.0f}s)", flush=True)

    y = np.concatenate(all_y)
    s_p = np.concatenate(all_s_p)
    s_a = np.concatenate(all_s_a)
    s_d = np.concatenate(all_s_d)
    pooled = dict(
        auc_ours=roc_auc(y, s_p),
        auc_appear=roc_auc(y, s_a),
        auc_framediff=roc_auc(y, s_d),
        n_frames=int(len(y)),
        n_abnormal=int(y.sum()),
    )
    aucs = {k: [r[k] for r in per_video] for k in ("auc_ours", "auc_appear", "auc_framediff")}
    summary = {k: dict(median=float(np.nanmedian(v)), min=float(np.nanmin(v)),
                       max=float(np.nanmax(v))) for k, v in aucs.items()}

    result = dict(
        seed=SEED, patch=PATCH, step=STEP, acf_bins=ACF_BINS, top_q=TOP_Q,
        acf_weight=ACF_W, n_train_frames_per_vol=N_TRAIN_FRAMES_PER_VOL,
        n_bank_patches=int(len(bank_p)),
        pooled=pooled, per_video=per_video, per_video_summary=summary,
        elapsed_s=round(time.time() - t0, 1),
    )
    with open(OUT / "e13_avenue_anomaly.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(dict(pooled=pooled, summary=summary), indent=1))
    print(f"saved -> {OUT/'e13_avenue_anomaly.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
