"""e11_per_pitch.py — 逐 pitch 的分支分类准确率（裁决"两个周期"叙事）

若只有 p|96 的 5 组停在 chance、其余 11 组显著高于 chance，
则"可见周期数 < 2 就不可辨识"被自家数据反向证伪。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import e1_baselines as B
import e1_pavt as P

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_FRAI论文" / "04-实验" / "results"
W = 96


def main() -> None:
    ckpt = OUT / "pavt_lite.pt"
    assert ckpt.exists(), "缺少 pavt_lite.pt"
    model = P.PAVT()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(123)
    rows = []
    print(f"{'p':>4} {'96%p':>5} {'N=96/p':>6} {'branch acc':>11} {'n':>4}")
    for p in B.PITCHES:
        acc, n = 0, 0
        for _ in range(40):
            d = float(rng.uniform(0, 2 * p))
            I0, I1 = B.render(p, d, rng)
            fr = torch.from_numpy(np.stack([I0, I1])[None, :, None].astype(np.float32))
            pt = torch.tensor([p], dtype=torch.float32)
            with torch.no_grad():
                phase, logits, _ = model(fr, pt)
                k_hat = int(logits.argmax(-1))
            k_true = int(np.clip(np.floor(d / p), 0, P.K_MAX - 1))
            acc += int(k_hat == k_true)
            n += 1
        rows.append({"pitch": p, "w_mod_p": W % p, "periods": W / p,
                     "acc": acc / n, "n": n})
        print(f"{p:4.0f} {W % p:5d} {W/p:6.1f} {acc/n:11.3f} {n:4d}")

    div = [r["acc"] for r in rows if r["w_mod_p"] == 0]
    nodiv = [r["acc"] for r in rows if r["w_mod_p"] != 0]
    print(f"\n整除组 (n={len(div)}) 平均 {np.mean(div):.3f} | "
          f"非整除组 (n={len(nodiv)}) 平均 {np.mean(nodiv):.3f}")
    (OUT / "e11_per_pitch.json").write_text(
        json.dumps({"rows": rows,
                    "mean_divisible": float(np.mean(div)),
                    "mean_nondivisible": float(np.mean(nodiv))},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写入 {OUT/'e11_per_pitch.json'}")


if __name__ == "__main__":
    main()
