"""e2b_adapt.py — 域内自监督校准：用真实视频的跨基线一致性作代理标签

问题：合成数据训练的可观测性头在真实视频上零迁移（Spearman +0.006）。
本实验检验：**特征里是否含有信息**？即在目标域用代理标签重训后，
能否在留出视频上正确排序。

注意（写论文必须交代）：代理标签本身就是一致性度量，因此这是
"能否学会代理指标"，不是"能否预测真误差"。它能区分两种失败：
  · 特征里没信息 → 重训也无效（应回到方法层面）
  · 特征里有信息但域偏移 → 重训有效（说明需要目标域校准）

划分：**按视频分组**（避免同一视频的相邻帧泄漏到测试集）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from e1_hybrid import ObsNet

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "09_FRAI论文" / "04-实验" / "results"
SEED = 20261010


def curve(scores, errs, fracs=(1.0, 0.8, 0.6, 0.4)):
    idx = np.argsort(scores)
    e = errs[idx]
    return {f: float(np.mean(e[:max(1, int(round(len(e) * f)))])) for f in fracs}


def train_obs(X, y, steps=600, bs=32, lr=3e-3):
    torch.manual_seed(SEED)
    m = ObsNet(X.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (min(bs, len(xt)),))
        loss = F.l1_loss(m(xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def main():
    d = json.loads((RES / "e2_real.json").read_text(encoding="utf-8"))
    rows = [r for r in d["rows"] if r.get("feat") and np.isfinite(r["score"])]
    print(f"可用样本 {len(rows)}，视频 {len({r['video'] for r in rows})} 段")

    vids = sorted({r["video"] for r in rows})
    rng = np.random.default_rng(SEED)
    rng.shuffle(vids)
    ntr = max(1, int(round(0.7 * len(vids))))
    tr_v, te_v = set(vids[:ntr]), set(vids[ntr:])

    tr = [r for r in rows if r["video"] in tr_v]
    te = [r for r in rows if r["video"] in te_v]
    print(f"训练视频 {len(tr_v)}（{len(tr)} 样本）| 测试视频 {len(te_v)}（{len(te)} 样本）")

    Xtr = np.stack([r["feat"] for r in tr]).astype(np.float32)
    ytr = np.array([r["incons"] for r in tr], dtype=np.float32)
    Xte = np.stack([r["feat"] for r in te]).astype(np.float32)
    yte = np.array([r["incons"] for r in te], dtype=np.float32)
    zs = np.array([r["score"] for r in te], dtype=np.float32)
    pr = np.array([r["ratio"] for r in te], dtype=np.float32)

    m = train_obs(Xtr, ytr)
    with torch.no_grad():
        adapt = m(torch.from_numpy(Xte)).numpy()

    print("\n=== 留出视频上的排序能力（Spearman with 代理误差）===")
    for name, s in (("zero-shot (synthetic)", zs), ("peak-ratio heuristic", -pr), ("adapted (ours)", adapt)):
        if np.std(s) > 1e-9:
            rho, p = spearmanr(s, yte)
            print(f"{name:24s} rho={rho:+.3f} (p={p:.1e})")
        else:
            print(f"{name:24s} 常数，无法计算")

    print("\n=== 风险—覆盖率（误差=代理误差）===")
    res = {"random": curve(np.random.default_rng(0).random(len(te)), yte),
           "zero-shot": curve(zs, yte),
           "peak-ratio": curve(-pr, yte),
           "adapted": curve(adapt, yte)}
    print(f"{'policy':14s} {'@100%':>8s} {'@80%':>8s} {'@60%':>8s} {'@40%':>8s}")
    for k, v in res.items():
        print(f"{k:14s} {v[1.0]:8.3f} {v[0.8]:8.3f} {v[0.6]:8.3f} {v[0.4]:8.3f}")

    (RES / "e2b_adapt.json").write_text(json.dumps(
        {"n_tr": len(tr), "n_te": len(te), "n_video_tr": len(tr_v), "n_video_te": len(te_v),
         "curves": res,
         "rows_te": [{"video": r["video"], "incons": float(y),
                      "score_zs": float(z), "score_adapted": float(a), "ratio": float(pv)}
                     for r, y, z, a, pv in zip(te, yte, zs, adapt, pr)]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {RES/'e2b_adapt.json'}")


if __name__ == "__main__":
    main()
