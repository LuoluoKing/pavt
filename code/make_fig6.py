"""make_fig6.py — Fig.6：真实扶梯视频上的风险—覆盖率（域内校准前后对比）"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "09_FRAI论文" / "04-实验" / "results" / "e2b_adapt.json"
OUT = ROOT / "09_FRAI论文" / "03-图表"

BLUE, VERM, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def main() -> None:
    d = json.loads(RES.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    order = [("zero-shot", "synthetic-trained (no adaptation)", ORANGE, "--"),
             ("peak-ratio", "peak-to-sidelobe ratio", GREY, ":"),
             ("random", "random abstention", VERM, "-."),
             ("adapted", "calibrated on target domain (ours)", GREEN, "-")]
    for key, label, color, ls in order:
        v = d["curves"][key]
        cov = np.array([1.0, 0.8, 0.6, 0.4])
        y = np.array([v[f] for f in ("1.0", "0.8", "0.6", "0.4")])
        ax.plot(cov, y, ls, color=color, lw=1.6 if key == "adapted" else 1.2,
                marker="o", ms=3, label=label)
    ax.invert_xaxis()
    ax.set_xlabel("coverage (fraction of measurements reported)")
    ax.set_ylabel("two-baseline disagreement (px/frame)")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    ax.set_title("Real escalator video, 10 held-out clips", loc="left",
                 fontsize=9, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig6_real_coverage.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("saved fig6_real_coverage")


if __name__ == "__main__":
    main()
