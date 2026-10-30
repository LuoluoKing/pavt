"""e14_stats.py — E14：真实视频实验的统计完善（回应审稿 M12 与统计意见）

四个分析，全部基于 e2_real.json / e2b_adapt.json 已保存的逐测量明细：

A. 视频级聚类 bootstrap CI
   测量在同一视频内相关，逐测量 i.i.d. 自举会低估方差。
   以视频为重抽样单元（B=10000）给出以下统计量的 95% CI：
   - e2_real pooled：rho(score, incons)、rho(-ratio, incons)
   - e2b 留出集：rho(adapted, incons)、rho(zero-shot, incons)

B. 偏相关：把可靠性分数与位移幅值解耦（M12）
   - m1 = |d1|（帧间位移幅值），相对量 m1/p
   - 秩变换后用线性残差法计算偏相关：
     rho(score, incons | m1/p)
   - 对照：rho(score, incons) 与 rho(m1/p, incons)、rho(score, m1/p)

C. 按相对位移分层的三层风险-覆盖率
   层：m1/p ∈ [0,0.25), [0.25,0.5), [0.5,1.0)（并报告 [1.0,∞) 剩余层）
   每层内报告 ours / peak-ratio / random 在 100%/60% 覆盖率下的不一致度。

D. M12 相关矩阵 + 视频级 ICC
   - 五组 Spearman：score↔incons、score↔pitch、pitch↔incons、
     score↔m1/p、acf↔incons（pooled 264）
   - ICC(1,1)（one-way ANOVA 组间方差占比）：incons 与 score 的视频级聚类程度
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parent / "results"
B_BOOT = 10000
SEED = 20261010


def rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def cluster_boot_ci(values: np.ndarray, groups: np.ndarray, stat_fn, b: int = B_BOOT,
                    seed: int = SEED) -> dict:
    """视频级聚类 bootstrap。stat_fn: (values[idx], groups_idx_values) -> float"""
    rng = np.random.default_rng(seed)
    uv = np.unique(groups)
    by_v = {v: np.where(groups == v)[0] for v in uv}
    point = stat_fn(values, np.arange(len(values)))
    stats = []
    for _ in range(b):
        vs = rng.choice(uv, size=len(uv), replace=True)
        idx = np.concatenate([by_v[v] for v in vs])
        if len(np.unique(groups[idx])) < 2:
            continue
        stats.append(stat_fn(values, idx))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return dict(point=float(point), ci=[float(lo), float(hi)],
                n_boot=len(stats))


def rho_stat(which_a: int, which_b: int):
    """返回 stat_fn：从 values 矩阵取两列算 Spearman。"""
    def f(vals, idx):
        r, _ = spearmanr(vals[idx, which_a], vals[idx, which_b])
        return float(r)
    return f


def partial_corr_rank(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """秩变换 + 线性残差偏相关。"""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, c):
        A = np.stack([np.ones_like(c), c], axis=1)
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta

    ex = resid(rx, rz)
    ey = resid(ry, rz)
    return pearson(ex, ey)


def curve_at(scores: np.ndarray, errs: np.ndarray, frac: float) -> float:
    idx = np.argsort(scores)
    e = errs[idx]
    k = max(1, int(round(len(e) * frac)))
    return float(np.mean(e[:k]))


def random_curve_at(errs: np.ndarray, frac: float, seed: int = 0) -> float:
    s = np.random.default_rng(seed).random(len(errs))
    return curve_at(s, errs, frac)


def main():
    d2 = json.loads((RES / "e2_real.json").read_text(encoding="utf-8"))
    rows = [r for r in d2["rows"] if r.get("feat") and np.isfinite(r["score"])]
    db = json.loads((RES / "e2b_adapt.json").read_text(encoding="utf-8"))
    rows_te = db["rows_te"]

    vids = np.array([r["video"] for r in rows])
    incons = np.array([r["incons"] for r in rows], float)
    score = np.array([r["score"] for r in rows], float)
    ratio = np.array([r["ratio"] for r in rows], float)
    pitch = np.array([r["pitch"] for r in rows], float)
    acf = np.array([r["acf"] for r in rows], float)
    d1 = np.array([r["d1"] for r in rows], float)  # (n,2)
    m1 = np.hypot(d1[:, 0], d1[:, 1])
    m1p = m1 / np.maximum(pitch, 1e-6)
    n = len(rows)
    print(f"e2_real: {n} 测量 / {len(set(vids))} 视频 | m1/p 分位: "
          f"25%={np.quantile(m1p,0.25):.3f} 50%={np.quantile(m1p,0.5):.3f} 75%={np.quantile(m1p,0.75):.3f} max={m1p.max():.2f}")

    # ---------- A. 聚类 bootstrap CI ----------
    M = np.stack([score, incons, -ratio, m1p, pitch, acf], axis=1)  # 列 0..5
    A = dict(
        rho_score_incons=cluster_boot_ci(M, vids, rho_stat(0, 1)),
        rho_ratio_incons=cluster_boot_ci(M, vids, rho_stat(2, 1)),
    )
    print("\n[A] e2_real pooled（视频聚类 bootstrap, B=%d）" % B_BOOT)
    for k, v in A.items():
        print(f"  {k}: {v['point']:+.3f}  95%CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")

    tv = np.array([r["video"] for r in rows_te])
    ti = np.array([r["incons"] for r in rows_te], float)
    ta = np.array([r["score_adapted"] for r in rows_te], float)
    tz = np.array([r["score_zs"] for r in rows_te], float)
    B = dict(
        rho_adapted_incons=cluster_boot_ci(np.stack([ta, ti], 1), tv, rho_stat(0, 1)),
        rho_zeroshot_incons=cluster_boot_ci(np.stack([tz, ti], 1), tv, rho_stat(0, 1)),
    )
    print("[A] e2b 留出 10 视频")
    for k, v in B.items():
        print(f"  {k}: {v['point']:+.3f}  95%CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")

    # ---------- B. 偏相关 ----------
    Bsec = dict(
        rho_score_incons=float(spearmanr(score, incons)[0]),
        rho_score_m1p=float(spearmanr(score, m1p)[0]),
        rho_m1p_incons=float(spearmanr(m1p, incons)[0]),
        partial_score_incons_given_m1p=partial_corr_rank(score, incons, m1p),
    )
    print("\n[B] 偏相关（秩变换线性残差法）")
    for k, v in Bsec.items():
        print(f"  {k}: {v:+.3f}")

    # ---------- C. 分层风险-覆盖率 ----------
    edges = [0.0, 0.25, 0.5, 1.0, np.inf]
    C_layers = []
    print("\n[C] 按 m1/p 分层（误差=不一致度）")
    print(f"{'layer':16s} {'n':>4s} {'ours@1.0':>9s} {'ratio@1.0':>10s} {'ours@0.6':>9s} {'ratio@0.6':>10s} {'rnd@0.6':>8s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (m1p >= lo) & (m1p < hi)
        if m.sum() < 10:
            continue
        c = dict(
            layer=f"[{lo:.2f},{hi:.2f})" if np.isfinite(hi) else f"[{lo:.2f},inf)",
            n=int(m.sum()),
            ours_full=curve_at(score[m], incons[m], 1.0),
            ratio_full=curve_at(-ratio[m], incons[m], 1.0),
            ours_60=curve_at(score[m], incons[m], 0.6),
            ratio_60=curve_at(-ratio[m], incons[m], 0.6),
            random_60=random_curve_at(incons[m], 0.6, seed=0),
        )
        C_layers.append(c)
        print(f"{c['layer']:16s} {c['n']:4d} {c['ours_full']:9.3f} {c['ratio_full']:10.3f} "
              f"{c['ours_60']:9.3f} {c['ratio_60']:10.3f} {c['random_60']:8.3f}")

    # ---------- D. M12 相关矩阵 + ICC ----------
    D = dict(
        rho_score_incons=float(spearmanr(score, incons)[0]),
        rho_score_pitch=float(spearmanr(score, pitch)[0]),
        rho_pitch_incons=float(spearmanr(pitch, incons)[0]),
        rho_score_m1p=float(spearmanr(score, m1p)[0]),
        rho_acf_incons=float(spearmanr(acf, incons)[0]),
    )

    def icc11(x: np.ndarray, g: np.ndarray) -> float:
        uv = np.unique(g)
        k = len(uv)
        grand = x.mean()
        ss_b = sum(len(x[g == v]) * (x[g == v].mean() - grand) ** 2 for v in uv)
        ss_w = sum(((x[g == v] - x[g == v].mean()) ** 2).sum() for v in uv)
        n0 = len(x)
        k0 = n0 / k  # 平均组大小
        ms_b = ss_b / (k - 1)
        ms_w = ss_w / (n0 - k)
        return float((ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w)) if ms_b + (k0 - 1) * ms_w > 0 else 0.0

    D["icc_video_incons"] = icc11(incons, vids)
    D["icc_video_score"] = icc11(score, vids)
    D["icc_video_m1p"] = icc11(m1p, vids)
    print("\n[D] M12 相关与 ICC")
    for k, v in D.items():
        print(f"  {k}: {v:+.3f}")

    out = dict(
        seed=SEED, n=n, b_boot=B_BOOT,
        A_cluster_ci=dict(e2_real=A, e2b_heldout=B),
        B_partial=Bsec,
        C_layers=C_layers,
        D_m12=D,
    )
    (RES / "e14_stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {RES/'e14_stats.json'}")


if __name__ == "__main__":
    main()
