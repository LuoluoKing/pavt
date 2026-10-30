"""e18_rule_only.py — E10：免训练边界残差规则命名分支 vs 纯弃权（回应审稿人 E10）

审稿人指出：既然非整除 pitch 上免训练规则能到 0.73，逻辑上更自然的系统是
"非整除用规则命名 + 整除弃权"，而非全文弃权。本实验补这个对照。

协议
----
- 数据：与 e1_hybrid 完全相同的 3,072 对合成样本（同 seed 复现），
  取 30% 留出（922 对，与表 6/表 7 同一划分）。
- 策略对比（误差 = |Δ̂ − Δ_true| / p）：
  A. 纯弃权（当前系统）：全报告时用相位 φ̂，不命名分支（即论文口径）
  B. rule-only：非整除 pitch 用边界残差规则命名分支，整除 pitch 弃权
  C. rule + 选择器：规则先定分支，选择器再决定是否报告
- 指标：HAR（分支错误率）、MAE@100%、在覆盖率 {1.0,0.8,0.6,0.4} 下的误差
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import e1_baselines as B
import e1_hybrid as H
from e15c_simple_bound import boundary_classify, shift_x

OUT = Path(__file__).resolve().parent / "results"
W = 96
SEED = 20261010
DIVISIBLE = {8, 12, 16, 24, 32}


def rebuild_pairs():
    """按 e1_hybrid.build 的随机流复现样本，同时保留图像对与真值。"""
    rng = np.random.default_rng(SEED)
    degs = [("none", {}), ("noise20", {"noise": 20}), ("noise60", {"noise": 60}),
            ("occl15", {"occl": 0.15}), ("occl30", {"occl": 0.30}), ("jpeg30", {"jpeg_q": 30})]
    p_list, d_list, phi_list, deg_list = [], [], [], []
    pairs = []
    for regime in ("small", "large"):
        for p in B.PITCHES:
            lo, hi = (0.0, p / 2.0) if regime == "small" else (p / 2.0, 2.0 * p)
            for dname, deg in degs:
                for _ in range(16):
                    d = float(rng.uniform(lo, hi))
                    I0, I1 = B.render(p, d, rng, **deg)
                    f, phi, ratio = H.features(I0, I1, p)
                    pairs.append((I0, I1))
                    p_list.append(p); d_list.append(d); phi_list.append(phi); deg_list.append(dname)
    return pairs, np.array(p_list), np.array(d_list), np.array(phi_list), deg_list


def main():
    print("复现 3,072 对样本…", flush=True)
    pairs, P, D, PHI, DEG = rebuild_pairs()
    n = len(pairs)
    assert n == 3072, n

    idx = np.random.default_rng(SEED).permutation(n)
    te = idx[int(0.7 * n):]
    print(f"留出 {len(te)} 对", flush=True)

    # 对留出样本应用边界残差规则
    k_rule = np.full(len(te), -1, dtype=int)
    for i, j in enumerate(te):
        I0, I1 = pairs[j]
        p = float(P[j])
        phi = float(PHI[j]) % p
        if int(p) in DIVISIBLE:
            continue  # 弃权
        k_hat, _ = boundary_classify(I0, I1, p, phi)
        k_rule[i] = k_hat

    p_te, d_te, phi_te = P[te], D[te], PHI[te]
    # A. 纯相位（当前系统口径：不命名分支）
    err_phase = np.abs(phi_te - d_te) / p_te
    # B. rule-only：非整除用规则，整除弃权
    rep = k_rule >= 0
    d_hat = phi_te.copy()
    d_hat[rep] = phi_te[rep] + k_rule[rep] * p_te[rep]
    err_rule = np.abs(d_hat - d_te) / p_te

    k_true = np.floor(d_te / p_te).astype(int)
    har_phase = float(np.mean(np.array([B.branch(ph, p) != B.branch(d, p) for ph, d, p in zip(phi_te, d_te, p_te)])))
    har_rule_rep = float(np.mean(k_rule[rep] != np.clip(k_true[rep], 0, 2))) if rep.any() else float('nan')
    # 含弃权的 HAR：弃权不算错，只统计已报告的
    print(f"\n覆盖率（rule 可命名比例）: {rep.mean():.3f} "
          f"({rep.sum()}/{len(te)}; 整除 pitch 弃权)")
    print(f"HAR — 纯相位（全报告）: {har_phase:.3f}")
    print(f"HAR — rule（仅已报告）: {har_rule_rep:.3f}")
    print(f"MAE@100% — 纯相位: {err_phase.mean():.4f}")
    print(f"MAE@100% — rule（弃权者按相位计）: {err_rule.mean():.4f}")

    # 覆盖率曲线：rule 报告的非整除子集，误差按报告集统计
    rows = []
    for name, err in (("phase-only", err_phase), ("rule-only", err_rule)):
        rows.append({"policy": name, "mean_err_all": float(err.mean()),
                     "median": float(np.median(err))})
    # 仅在 rule 报告的样本上比较
    print(f"\n仅在 rule 可命名子集上（n={rep.sum()}）：")
    print(f"  纯相位 MAE: {err_phase[rep].mean():.4f} | rule MAE: {err_rule[rep].mean():.4f}")
    print(f"  纯相位 HAR: {np.mean([B.branch(ph,p)!=B.branch(d,p) for ph,d,p in zip(phi_te[rep],d_te[rep],p_te[rep])]):.3f} | rule HAR: {har_rule_rep:.3f}")

    (OUT / "e18_rule_only.json").write_text(json.dumps({
        "n_heldout": int(len(te)), "n_divisible_pitch": len(DIVISIBLE),
        "rule_coverage": float(rep.mean()),
        "har_phase_all": har_phase, "har_rule_reported": har_rule_rep,
        "mae_phase_all": float(err_phase.mean()), "mae_rule_all": float(err_rule.mean()),
        "mae_phase_on_reported": float(err_phase[rep].mean()) if rep.any() else None,
        "mae_rule_on_reported": float(err_rule[rep].mean()) if rep.any() else None,
        "rows": rows,
        "note": "A=纯相位（当前系统不命名分支）；B=rule-only（非整除用边界残差，整除弃权）",
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"\n已写入 {OUT/'e18_rule_only.json'}")


if __name__ == "__main__":
    main()
