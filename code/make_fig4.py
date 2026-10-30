"""make_fig4.py — 用 e1_baselines.json / e1_pavt.json 的真实数据画 Fig.4

两联图：
  (A) 位移 MAE 随节距变化（小位移 / 大位移两个 regime）
  (B) 谐波歧义率 HAR 随节距变化
只画 JSON 中真实存在的条目；PAVT 若尚无结果则不画。
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "09_FRAI论文" / "04-实验" / "results"
OUT = ROOT / "09_FRAI论文" / "03-图表"

BLUE, VERM, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def series(cells, method, regime, key, deg_filter=lambda c: c["degradation"] == {}):
    """按节距聚合（无退化条件）。"""
    acc = defaultdict(list)
    for c in cells:
        if c["regime"] != regime or not deg_filter(c):
            continue
        acc[c["pitch"]].append(c[key][method])
    ps = sorted(acc)
    return np.array(ps), np.array([np.mean(acc[p]) for p in ps])


def main() -> None:
    base = load("e1_baselines.json")
    if base is None:
        print("缺少 e1_baselines.json"); return
    cells = base["cells"]
    # SHOW_PAVT=1 时才画 PAVT（模型重训期间保持关闭，避免混入过期曲线）
    pavt = load("e1_pavt.json") if os.environ.get("SHOW_PAVT") == "1" else None

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex=True)
    methods = [("global", "global correlation", GREY, "-"),
               ("band", "prior-constrained", BLUE, "--"),
               ("lk1d", "1-D Lucas–Kanade", ORANGE, "-.")]

    for col, regime in enumerate(("small", "large")):
        for row, key in enumerate(("mae", "har")):
            ax = axes[row, col]
            for m, label, color, ls in methods:
                ps, ys = series(cells, m, regime, key)
                ax.plot(ps, ys, ls, color=color, lw=1.3, marker="o", ms=2.5, label=label)
            if pavt is not None:
                pc = [c for c in pavt["cells"] if c["regime"] == regime and c["degradation"] == {}]
                if pc:
                    xs = sorted({c["pitch"] for c in pc})
                    ys = [np.mean([c[key] for c in pc if c["pitch"] == x]) for x in xs]
                    ax.plot(xs, ys, color=GREEN, lw=1.8, marker="s", ms=3, label="PAVT (ours)")
            ax.set_xscale("log")
            ax.set_xticks([8, 16, 32, 64])
            ax.set_xticklabels(["8", "16", "32", "64"])
            ax.minorticks_off()
            if key == "mae":
                ax.set_yscale("log")
                ax.set_ylabel("displacement MAE (px)")
            else:
                ax.set_ylabel("harmonic ambiguity rate")
                ax.set_ylim(-0.03, 1.03)
            if row == 1:
                ax.set_xlabel("pitch $p$ (px)")
        axes[0, col].set_title("A  small motion $|\\Delta|<p/2$" if col == 0
                               else "B  large motion $|\\Delta|\\in[p/2,2p)$",
                               loc="left", fontsize=9, fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_measurement.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("saved fig4_measurement")


if __name__ == "__main__":
    main()
