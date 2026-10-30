"""verify_E2.py — 独立复核 reviewer-experiment 的 E2 指控

指控：264 条真实测量中 70% 的 8 帧基线落在 ±p/2 之外（跨基线缠绕），
主统计 ρ=+0.27 主要来自这部分被污染的子样本。
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

R = Path('../04-实验/results')

d = json.loads((R / 'e2_real.json').read_text(encoding='utf-8'))
rows = [r for r in d['rows'] if r.get('feat') and np.isfinite(r['score'])]
print('样本:', len(rows))

score = np.array([r['score'] for r in rows], float)
incons = np.array([r['incons'] for r in rows], float)
d1 = np.array([r['d1'] for r in rows], float)
pitch = np.array([r['pitch'] for r in rows], float)
m1 = np.hypot(d1[:, 0], d1[:, 1])
m1p = m1 / np.maximum(pitch, 1e-6)

# 1. 缠绕判定：2|d1| > p/2  <=>  |d1|/p > 0.25
wrap = m1p > 0.25
print(f'\n|d1|/p > 0.25 的比例: {wrap.sum()}/{len(rows)} = {wrap.mean():.3f}')

# 2. 分层 ρ
for name, mask in (('无缠绕 (|d1|/p<=0.25)', ~wrap), ('缠绕 (|d1|/p>0.25)', wrap), ('全部', np.ones(len(rows), bool))):
    n = mask.sum()
    rho, p = spearmanr(score[mask], incons[mask])
    print(f'{name:26s} n={n:3d}  rho={rho:+.3f}  p={p:.4f}')

# 3. 更严格的无缠绕：2|d1|/p < 0.5 等价于 |d1|/p < 0.25，同上
#    再加一个更保守的：|d1|/p < 0.15
for thr in (0.10, 0.15, 0.20, 0.25, 0.30):
    mask = m1p < thr
    if mask.sum() >= 15:
        rho, p = spearmanr(score[mask], incons[mask])
        print(f'  thr={thr:.2f}: n={mask.sum():3d} rho={rho:+.3f} p={p:.4f}')

# 4. G 敏感性（e9_decisive.json）
try:
    e9 = json.loads((R / 'e9_decisive.json').read_text(encoding='utf-8'))
    print('\n=== e9_decisive.json 结构 ===')
    print(list(e9.keys())[:10])
    for k, v in e9.items():
        if isinstance(v, dict) and any(str(g) in str(v) for g in ('1', '2', '4', '8')):
            print(f'  {k}: {json.dumps(v)[:300]}')
except Exception as e:
    print('e9 读取失败:', e)

# 5. 无缠绕子集的聚类 CI（按视频）
def cluster_boot(mask, B=5000, seed=20261010):
    vids = np.array([r['video'] for r in rows])[mask]
    s = score[mask]; e = incons[mask]
    uv = np.unique(vids)
    by = {v: np.where(vids == v)[0] for v in uv}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        vs = rng.choice(uv, size=len(uv), replace=True)
        idx = np.concatenate([by[v] for v in vs])
        if len(np.unique(vids[idx])) < 2:
            continue
        stats.append(spearmanr(s[idx], e[idx])[0])
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)), len(uv)

lo, hi, nv = cluster_boot(~wrap)
print(f'\n无缠绕子集聚类 bootstrap 95% CI: [{lo:+.3f}, {hi:+.3f}] ({nv} 个视频簇)')
lo2, hi2, nv2 = cluster_boot(wrap)
print(f'缠绕子集聚类 bootstrap 95% CI:   [{lo2:+.3f}, {hi2:+.3f}] ({nv2} 个视频簇)')
