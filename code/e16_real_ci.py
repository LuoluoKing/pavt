"""e16_real_ci.py — M4/N2 补充：表 5（real transfer）覆盖率列的视频聚类 bootstrap CI

对 e2b_adapt.json 的 80 条留出测量（10 视频），按视频为单元重抽样（B=10000），
重新计算各策略在 100%/80%/60%/40% 覆盖率下的不一致度，给出 95% CI。
同时给出 adapted − zero-shot 的差值 CI（配对比较）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
B_BOOT = 10000
SEED = 20261010
FRACS = (1.0, 0.8, 0.6, 0.4)


def curve_at(scores: np.ndarray, errs: np.ndarray, frac: float) -> float:
    idx = np.argsort(scores)
    e = errs[idx]
    k = max(1, int(round(len(e) * frac)))
    return float(np.mean(e[:k]))


def boot_ci(vals, groups, score_col, b=B_BOOT, seed=SEED):
    """vals: dict of arrays (errs, score_a, score_b...)；按视频重抽样后重算曲线。"""
    rng = np.random.default_rng(seed)
    uv = np.unique(groups)
    by_v = {v: np.where(groups == v)[0] for v in uv}
    point = {f: curve_at(vals[score_col], vals["errs"], f) for f in FRACS}
    stats = {f: [] for f in FRACS}
    for _ in range(b):
        vs = rng.choice(uv, size=len(uv), replace=True)
        idx = np.concatenate([by_v[v] for v in vs])
        s, e = vals[score_col][idx], vals["errs"][idx]
        for f in FRACS:
            stats[f].append(curve_at(s, e, f))
    ci = {f: [float(np.quantile(stats[f], 0.025)), float(np.quantile(stats[f], 0.975))]
          for f in FRACS}
    return point, ci


def main():
    d = json.loads((RES / "e2b_adapt.json").read_text(encoding="utf-8"))
    rows = d["rows_te"]
    vids = np.array([r["video"] for r in rows])
    errs = np.array([r["incons"] for r in rows], float)
    zs = np.array([r["score_zs"] for r in rows], float)
    ad = np.array([r["score_adapted"] for r in rows], float)
    pr = np.array([-r["ratio"] for r in rows], float)
    rnd = np.random.default_rng(0).random(len(rows))

    out = {"n": len(rows), "n_video": len(set(vids)), "b_boot": B_BOOT, "policies": {}}
    print(f"留出集：{len(rows)} 测量 / {len(set(vids))} 视频")
    for name, s in (("zero-shot", zs), ("peak-ratio", pr), ("adapted", ad), ("random", rnd)):
        point, ci = boot_ci({"errs": errs, "s": s}, vids, "s")
        out["policies"][name] = dict(point=point, ci=ci)
        line = "  ".join(f"{f:.0%}:{point[f]:.3f}[{ci[f][0]:.3f},{ci[f][1]:.3f}]" for f in FRACS)
        print(f"  {name:12s} {line}")

    # adapted − zero-shot 配对差值
    rng = np.random.default_rng(SEED + 1)
    uv = np.unique(vids)
    by_v = {v: np.where(vids == v)[0] for v in uv}
    point_d = {f: curve_at(ad, errs, f) - curve_at(zs, errs, f) for f in FRACS}
    diffs = {f: [] for f in FRACS}
    for _ in range(B_BOOT):
        vs = rng.choice(uv, size=len(uv), replace=True)
        idx = np.concatenate([by_v[v] for v in vs])
        for f in FRACS:
            diffs[f].append(curve_at(ad[idx], errs[idx], f) - curve_at(zs[idx], errs[idx], f))
    ci_d = {f: [float(np.quantile(diffs[f], 0.025)), float(np.quantile(diffs[f], 0.975))]
            for f in FRACS}
    out["adapted_minus_zeroshot"] = dict(point=point_d, ci=ci_d)
    line = "  ".join(f"{f:.0%}:{point_d[f]:+.3f}[{ci_d[f][0]:+.3f},{ci_d[f][1]:+.3f}]" for f in FRACS)
    print(f"  {'adapted−zero':12s} {line}")

    (RES / "e16_real_ci.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n已写入 {RES/'e16_real_ci.json'}")


if __name__ == "__main__":
    main()
