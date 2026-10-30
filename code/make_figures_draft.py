"""make_figures_draft.py — 生成 FRAI 初稿的三张概念图（Fig.1 歧义 / Fig.2 架构 / Fig.3 相位对齐网格）

设计约束：Okabe-Ito 色盲友好；无渐变/阴影/装饰；字号可缩放；输出矢量 PDF + PNG 预览。
三张图均为**概念示意图**，不承载实验数值（数值图由实验脚本生成）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]   # 04-实验 -> 09_FRAI论文 -> 项目根
OUT = ROOT / "09_FRAI论文" / "03-图表"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, VERM, GREEN, ORANGE, PURPLE, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#7F7F7F")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Fig.1 歧义
def fig_ambiguity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # (A) 周期相关曲面
    ax = axes[0]
    p, d = 1.0, 0.22
    x = np.linspace(-2.6, 2.6, 2000)
    y = np.zeros_like(x)
    for k in range(-3, 4):
        y += np.exp(-((x - d - k * p) ** 2) / (2 * 0.055 ** 2))
    ax.plot(x, y, color=BLUE, lw=1.4)
    ax.axhline(0, color="black", lw=0.6)
    for k in (-1, 0, 1):
        ax.plot([d + k * p], [1.0], marker="v", ms=6, color=VERM)
    ax.annotate("true branch", xy=(d, 1.02), xytext=(d - 0.05, 1.28),
                fontsize=8, color=VERM, ha="center")
    ax.annotate("$\\Delta + kp$", xy=(d + 1.0, 1.02), xytext=(d + 1.05, 1.28),
                fontsize=8, color=VERM, ha="center")
    ax.set_xlabel("candidate displacement $\\delta$ (periods)")
    ax.set_ylabel("correlation $C(\\delta)$")
    ax.set_title("A  Equivalent peaks", loc="left", fontsize=9.5, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(0, 1.45)

    # (B) 阶跃式误差（示意）
    ax = axes[1]
    cond = np.arange(1, 11)
    err = np.array([0.3, 0.3, 0.3, 1.3, 1.3, 1.3, 1.3, 2.3, 2.3, 2.3])
    ax.step(cond, err, where="mid", color=VERM, lw=1.4)
    ax.plot(cond, err, "o", ms=3, color=VERM)
    for lev in (0, 1, 2):
        ax.axhline(lev, color=GREY, lw=0.4, ls=":")
    ax.set_xlabel("degradation level")
    ax.set_ylabel("error (periods)")
    ax.set_title("B  Step-like failure", loc="left", fontsize=9.5, fontweight="bold")
    ax.set_ylim(-0.25, 2.8)

    fig.tight_layout()
    save(fig, "fig1_ambiguity")


# --------------------------------------------------------------- Fig.2 架构
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 3.4))
    ax.set_xlim(0, 102); ax.set_ylim(0, 34); ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", fs=8.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
                                    fc=fc, ec="black", lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     lw=1.0, color="black", shrinkA=0, shrinkB=0))

    box(1, 15, 12, 7, "frame pair\n$I_t,\\,I_{t+1}$")
    box(16, 21, 17, 8, "2-D radial ACF\n$\\rightarrow$ pitch $\\hat p$", fc="#F4F7FB")
    box(16, 7, 17, 11, "phase stage\ncorr. on $|\\delta|\\leq\\hat p/2$\n+ parabola\n(sub-pixel)", fc="#EAF3EA")
    box(37, 17, 22, 12, "reliability head\nsurface $\\rightarrow$ tokens\n+ pitch, peak, degradation\n$\\rightarrow$ transformer\n$\\rightarrow \\hat e$", fc="#EAF3EA")
    box(63, 21, 15, 7, "decision\n$\\hat e \\leq \\tau$ ?", fc="#FFF4E5")
    box(63, 7, 15, 7, "report $\\hat\\Delta=\\hat\\phi$", fc="#FFF4E5", bold=True)
    box(82, 21, 17, 7, "abstain\n(no measurement)", fc="#F2F2F2")

    for a in [(13, 19, 16, 25), (13, 18, 16, 12), (33, 25, 37, 24),
              (33, 12, 37, 20), (59, 24, 63, 24.5)]:
        arrow(*a)
    arrow(70.5, 21, 70.5, 14)
    arrow(78, 24.5, 82, 24.5)

    ax.text(51, 2.0, "Trained with an $L_1$ loss on the realised error (synthetic) or on two-baseline "
                     "disagreement (target domain); no annotation required for recalibration.",
            ha="center", fontsize=7.6, color="#333333")
    save(fig, "fig2_architecture")


# --------------------------------------------------------- Fig.3 相位对齐网格
def fig_lattice() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    p = 20.0
    W = 240

    def stripe_bg(ax, shift=0.0, title=""):
        for i in range(-1, 13):
            x = i * p + shift
            ax.add_patch(Rectangle((x, 0), p / 2, 80, fc="#DDDDDD", ec="none"))
        ax.set_xlim(0, W); ax.set_ylim(0, 80); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")

    ax = axes[0]
    stripe_bg(ax, 0, "A  Structure (period $p$)")
    ax.annotate("", xy=(0, 88), xytext=(p, 88), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(p / 2, 92, "$p$", ha="center", fontsize=8)

    ax = axes[1]
    stripe_bg(ax, 0, "B  Border-anchored grid")
    xs = np.arange(0, W + 1, p)
    for x in xs:
        ax.plot([x, x], [0, 80], color=VERM, lw=0.8)
    ax.text(0.5, -10, "tokens move with the image when it shifts", fontsize=7.5, color=VERM, ha="left")

    ax = axes[2]
    shift = p
    stripe_bg(ax, shift, "C  Period-locked lattice (after shift $kp$)")
    for x in xs:
        ax.plot([x, x], [0, 80], color=GREEN, lw=0.8)
    ax.text(0.5, -10, "token set and phase coordinates are unchanged", fontsize=7.5, color=GREEN, ha="left")

    fig.tight_layout()
    save(fig, "fig3_lattice")


if __name__ == "__main__":
    fig_ambiguity()
    fig_architecture()
    fig_lattice()
