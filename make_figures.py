"""Headline figures, drawn only from the committed summaries in results/headline/.

Run: .venv/bin/python make_figures.py   (no simulation; takes a few seconds)
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "results/headline/m4-combined-freeze-rev5"
OUT = HERE / "results/headline"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#262624", "#6b6a64", "#e4e3dd"
LSM, NN = "#2a78d6", "#eb6834"          # validated categorical pair (CVD-safe)
BCJ = (26.138, 26.174)                  # Becker, Cheridito & Jentzen (2019), Table 1, 5D

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# Figure 1: paired neural - LSM test value against training budget (10 seeds each).
h2 = pd.read_csv(DATA / "H2.csv").sort_values("budget")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.axhspan(-0.05, 0.05, color=GRID, alpha=0.6, lw=0)
ax.axhline(0, color=MUTED, lw=1)
ax.text(1.02, 0.05, "±0.05 margin", transform=ax.get_yaxis_transform(), va="center",
        color=MUTED, fontsize=8.5)
x = range(len(h2))
ax.errorbar(x, h2["mean"], yerr=[h2["mean"] - h2["ci_low"], h2["ci_high"] - h2["mean"]],
            fmt="o", ms=8, color=NN, ecolor=NN, elinewidth=2, capsize=4)
for i, r in enumerate(h2.itertuples()):
    ax.annotate(f"{r.mean:+.3f}", (i, r.mean), xytext=(10, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)
ax.set_xticks(list(x), [f"{b // 1000}k" for b in h2["budget"]])
ax.set_xlabel("training paths per seed")
ax.set_ylabel("neural − LSM test value")
ax.set_title("Neural stopping vs Longstaff–Schwartz (mean, 95% CI over 10 seeds)",
             loc="left", fontsize=10.5, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "fig_neural_minus_lsm.png", dpi=160)

# Figure 2: primal-dual brackets for the two 50k policies against the published interval.
h3 = pd.read_csv(DATA / "H3.csv")
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.axvspan(*BCJ, color=GRID, alpha=0.7, lw=0)
ax.text(sum(BCJ) / 2, 1.62, "published 95% interval\n(Becker et al. 2019)", ha="center",
        va="bottom", color=MUTED, fontsize=8.5)
for y, (method, label, colour) in enumerate([("lsm", "LSM", LSM), ("sorted+payoff", "Neural", NN)]):
    r = h3[h3["method"] == method].iloc[0]
    lo, up = r["initial_value_mean"], r["upper_mean"]
    ax.plot([lo, up], [y, y], color=colour, lw=9, alpha=0.18, solid_capstyle="butt")
    for v, a, b in ((lo, r["initial_value_ci_low"], r["initial_value_ci_high"]),
                    (up, r["upper_ci_low"], r["upper_ci_high"])):
        ax.errorbar(v, y, xerr=[[v - a], [b - v]], fmt="o", ms=8, color=colour,
                    elinewidth=1.5, capsize=3)
    ax.annotate(f"gap {r['gap_mean']:.3f} ± {r['gap_half_width']:.3f}", (up, y),
                xytext=(0, 12), textcoords="offset points", ha="center", color=INK, fontsize=9)
ax.set_yticks([0, 1], ["LSM", "Neural"])
ax.set_ylim(-0.6, 2.1)
ax.set_xlabel("option value (lower bound ● … ● upper bound, 95% CIs)")
ax.set_title("How close to optimal? Andersen–Broadie bounds at 50k paths",
             loc="left", fontsize=10.5, color=INK)
style(ax)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(OUT / "fig_primal_dual_bounds.png", dpi=160)
print("wrote", OUT / "fig_neural_minus_lsm.png", "and", OUT / "fig_primal_dual_bounds.png")
