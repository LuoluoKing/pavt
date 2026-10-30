# PAVT — Period-Aware Video Transformers for Risk-Controlled Measurement of Periodic Structures

Code, result files and figure sources accompanying the manuscript

> **Jinhui Huang, Pengcheng Huang** (corresponding)
> Target venue: *Frontiers in Artificial Intelligence* (Pattern Recognition)

This repository reproduces every number reported in the manuscript. All experiments run
on a single CPU workstation (16 cores, 16 GB RAM); no GPU is used at any stage.

---

## What the paper claims, and where each claim comes from

| Claim | Experiment | Result file |
|---|---|---|
| Harmonic ambiguity is structural: classical estimators collapse above half a period | `e1_baselines.py` | `results/e1_baselines.json` |
| The branch is unidentifiable when the pitch divides the window, and a training-free rule beats a learned one elsewhere | `e15_fft_regen.py`, `e15b_verify.py`, `e15c_simple_bound.py` | `results/e15_*.json` |
| The selector halves the relative error at 60 % coverage | `e1_hybrid.py` | `results/e1_hybrid.json` |
| The reliability head does **not** transfer zero-shot to real escalator video; label-free recalibration restores it | `e2_real.py`, `e2b_adapt.py`, `e14_stats.py`, `e16_real_ci.py` | `results/e2_real.json`, `e2b_adapt.json`, `e14_stats.json`, `e16_real_ci.json` |
| Sub-pixel accuracy holds on real escalator texture (0.060 px median) | `e19_recovery_real.py` | `results/e19_recovery_real.json` |
| Naming the branch with the edge-residual rule is worse than abstaining under degradation | `e18_rule_only.py` | `results/e18_rule_only.json` |
| Feature ablation: the correlation surface carries the signal, not the peak statistics | `e17_ablation_5splits.py` | `results/e17_ablation_5splits.json` |
| CUHK Avenue frame-level AUC 0.844 | `e13_avenue_anomaly.py` | `results/e13_avenue_anomaly.json` |
| The synthetic-trained selector is anti-correlated with the proxy on held-out clips | `e2_real.py`, `e2b_adapt.py` | `results/e2_real.json` |

`docs/EXPERIMENTS.md` records the full experimental log, including the runs that failed
and the two bugs that changed conclusions (unloaded weights; a generator whose shifts were
not exact circular translations).

---

## Layout

```
code/         28 experiment scripts (e1 … e19, figure generators, verification)
results/      JSON output of every run reported in the paper
figures/      Figure 1–5 as vector PDF (Type-1 fonts; Type 3 removed)
data/         real_video_inventory.csv — per-video metadata of the 33 escalator clips
docs/         experiment log
```

The 33 escalator videos themselves are **not** redistributed: they are third-party content
from public video platforms. `data/real_video_inventory.csv` records, per video, the source
platform, duration, resolution, measured pitch, periodicity strength and per-video results,
so the set can be re-obtained. CUHK Avenue is publicly available from its authors.

---

## Reproducing

```bash
python -m pip install -r requirements.txt
cd code
python e1_baselines.py          # Table 2, Figure 3
python e1_hybrid.py             # Table 6, Figure 5, saves results/obs_synth.pt
python e2_real.py               # Table 4 (zero-shot), needs 01_数据/raw/*.mp4
python e2b_adapt.py             # Table 4 (recalibrated)
python e14_stats.py             # clustered intervals, partial correlation, ICC
python e16_real_ci.py           # coverage-level clustered intervals
python e17_ablation_5splits.py  # Table 7
python e15b_verify.py           # exactness of circular translation (no training)
python e15c_simple_bound.py     # training-free edge-residual rule (no training)
python e18_rule_only.py         # rule-only vs abstention
python e19_recovery_real.py     # sub-pixel recovery on real texture
python e15_fft_regen.py         # end-to-end branch probe, 3 conditions, ~3 h on CPU
python e13_avenue_anomaly.py    # Avenue AUC, needs the Avenue volumes, ~15 min
```

`e2_*` and `e19_*` expect the raw videos under `../01_数据/raw/`; `e13_*` expects the Avenue
dataset under `../01_data/`. Both paths are set at the top of the scripts.

Seeds are fixed (`20261010`) and every script writes a JSON file whose values are the ones
quoted in the paper. `verify_E2.py` re-derives the stratified correlation that the paper uses
to bound what the consistency proxy can certify.

---

## Environment

Python 3.13, NumPy, SciPy, OpenCV (headless), PyTorch (CPU build), Matplotlib, Pillow.
No GPU, no paid services, no private data.

## Citation

```bibtex
@article{huang2026pavt,
  title  = {Period-Aware Video Transformers for Risk-Controlled Measurement of Periodic Structures},
  author = {Huang, Jinhui and Huang, Pengcheng},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

Code and result files: MIT (see `LICENSE`). The figure PDFs may be reused under CC BY 4.0
with attribution to the manuscript.
