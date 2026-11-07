#!/usr/bin/env python3
"""Figures for the IEEE paper (IEEE single-column width = 3.5in). Grayscale-safe:
series are distinguished by line style and marker as well as color."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "latex" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
W1 = 3.5

STYLE = {
    "GBDT":           dict(color="#1f4e79", ls="-"),
    "ZScore":         dict(color="#c55a11", ls="--"),
    "LogReg":         dict(color="#548235", ls="-."),
    "MLP":            dict(color="#7030a0", ls=(0, (1, 1))),
    "Momentum":       dict(color="#7f7f7f", ls=(0, (5, 2, 1, 2))),
    "SeasonalFilter": dict(color="#000000", ls=(0, (3, 1))),
}
LABEL = {"ZScore": "Z-score composite", "SeasonalFilter": "Seasonal filter (all)",
         "GBDT": "GBDT", "LogReg": "Logistic reg.", "MLP": "MLP", "Momentum": "12-1 momentum"}

port = pd.read_csv(RES / "monthly_returns.csv", parse_dates=["date"]).set_index("date")

# Fig 2: cumulative log excess return vs EW universe
fig, ax = plt.subplots(figsize=(W1, 2.5))
for c, st in STYLE.items():
    ex = np.log1p(port[c] / 100) - np.log1p(port["EW"] / 100)
    ax.plot(ex.index, ex.cumsum() * 100, label=LABEL[c], **st)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Cum. log excess vs. EW (%)")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), handlelength=2.6)
ax.grid(alpha=0.25, lw=0.4)
fig.savefig(FIG / "cum_excess.pdf")
plt.close(fig)

# Fig 3: mean excess by calendar month
bym = pd.read_csv(RES / "excess_by_month.csv", index_col=0)
fig, ax = plt.subplots(figsize=(W1, 1.9))
x = np.arange(12)
wbar = 0.38
ax.bar(x - wbar / 2, bym["ZScore"], wbar, label="Z-score composite", color="#c55a11", edgecolor="k", lw=0.3)
ax.bar(x + wbar / 2, bym["GBDT"], wbar, label="GBDT", color="#1f4e79", edgecolor="k", lw=0.3, hatch="///")
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(x, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("Mean excess vs. EW (%/mo)")
ax.legend(frameon=False, loc="lower left")
ax.grid(axis="y", alpha=0.25, lw=0.4)
fig.savefig(FIG / "excess_by_month.pdf")
plt.close(fig)

# Fig 4: calibration -- win rate by score decile
fig, ax = plt.subplots(figsize=(W1, 1.9))
for n, mk in [("ZScore", "o"), ("GBDT", "s"), ("LogReg", "^")]:
    t = pd.read_csv(RES / f"calibration_{n}.csv")
    ax.plot(t["q"] + 1, t["win"] * 100, marker=mk, ms=3, label=LABEL[n],
            color=STYLE[n]["color"], ls=STYLE[n]["ls"])
ax.set_xticks(range(1, 11))
ax.set_xlabel("Score decile (1 = lowest)")
ax.set_ylabel("P(return > 0) (%)")
ax.set_ylim(50, 65)
ax.legend(frameon=False, ncol=3, loc="upper center")
ax.grid(alpha=0.25, lw=0.4)
fig.savefig(FIG / "calibration.pdf")
plt.close(fig)

# Fig 1: pipeline / evaluation protocol diagram
fig, ax = plt.subplots(figsize=(7.0, 1.25))
ax.axis("off")
boxes = [
    (0.00, "Monthly closes\n210 stocks\n1962–2025"),
    (0.205, "Point-in-time features\nknown at close of t−1\n(seasonal + controls)"),
    (0.41, "Eligibility (≥15 yrs)\n+ optional\nseasonal filter"),
    (0.615, "Ranker\nZ-score | LR\nGBDT | MLP"),
    (0.82, "Top-10 equal-weight\nportfolio, held for\nmonth t (−15 bp)"),
]
bw, bh, y0 = 0.175, 0.62, 0.30
for xpos, txt in boxes:
    ax.add_patch(plt.Rectangle((xpos, y0), bw, bh, fill=False, lw=0.7, transform=ax.transAxes))
    ax.text(xpos + bw / 2, y0 + bh / 2, txt, ha="center", va="center", fontsize=7.5, transform=ax.transAxes)
for (xa, _), (xb, _) in zip(boxes[:-1], boxes[1:]):
    ax.annotate("", xy=(xb, y0 + bh / 2), xytext=(xa + bw, y0 + bh / 2), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.7))
ax.text(0.5, 0.08, "Expanding-window training on months < January of test year; retrained annually",
        ha="center", fontsize=7.5, style="italic", transform=ax.transAxes)
fig.savefig(FIG / "pipeline.pdf")
plt.close(fig)

print("figures written to", FIG)
