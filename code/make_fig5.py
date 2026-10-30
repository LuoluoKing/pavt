"""make_fig5.py — Fig.5：风险—覆盖率曲线（真实数据，来自 e1_hybrid.json）"""
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
RES = ROOT / "09_FRAI论文" / "04-实验" / "results" / "e1_hybrid.json"
OUT = ROOT / "09_FRAI论文" / "03-图表"

BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def main() -> None:
    d = json.loads(RES.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    styles = {"peak-ratio": ("peak-ratio heuristic", GREY, "--"),
              "random": ("random abstention", VERM, ":"),
              "ours": ("learned observability (ours)", GREEN, "-")}
    for name, (label, color, ls) in styles.items():
        y = np.array(d["curves"][name])
        cov = 1.0 - np.arange(len(y)) / len(y)
        ax.plot(cov, y, ls, color=color, lw=1.5 if name == "ours" else 1.2, label=label)
    for cov in (0.6,):
        ax.axvline(cov, color=GREY, lw=0.6, ls="-.")
    r = d["results"]
    ax.annotate(f"{r['ours']['0.6']:.2f}", xy=(0.6, r["ours"]["0.6"]),
                xytext=(0.62, r["ours"]["0.6"] + 0.06), fontsize=8, color=GREEN)
    ax.set_xlabel("coverage (fraction of measurements reported)")
    ax.set_ylabel("relative displacement error")
    ax.invert_xaxis()
    ax.set_xlim(1.0, 0.35)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("Risk–coverage on the synthetic benchmark", loc="left",
                 fontsize=9, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_risk_coverage.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("saved fig5_risk_coverage")


if __name__ == "__main__":
    main()
