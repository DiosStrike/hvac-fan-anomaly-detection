# -*- coding: utf-8 -*-
"""HVAC 2.0 results summary: reads results/ and produces all figures + numbers.json + tables.md.

Usage: python code/analysis/make_figures.py        (run from the repo root)
Outputs: code/analysis/figures/fig*.png, numbers.json, tables.md, captions.md

Every number cited in REPORT.md comes from numbers.json (the key is the provenance identifier).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "code" / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- style
# CJK font: set explicitly with ordered fallback to avoid missing-glyph boxes
_CJK = ["Hiragino Sans GB", "PingFang HK", "PingFang SC", "Heiti TC", "STHeiti", "Arial Unicode MS",
        "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _CJK if f in _avail), None)
if FONT is None:
    print("⚠️ No CJK font found; CJK text in figures may render as boxes. Candidates:", _CJK, file=sys.stderr)
plt.rcParams.update({
    "font.family": [FONT] if FONT else ["sans-serif"],
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": "#898781", "ytick.color": "#898781", "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.labelcolor": "#52514e", "axes.labelsize": 10,
    "axes.titlecolor": "#0b0b0b", "axes.titlesize": 11.5, "axes.titleweight": "medium",
    "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})

# Fixed-order categorical colors, checked with the dataviz validator (light; adjacent + all-pairs PASS)
SCHEME_ORDER = ["full", "decoder_only", "encoder_only", "bottleneck_only"]
SCHEME_COLOR = {"full": "#2a78d6", "decoder_only": "#eb6834", "encoder_only": "#1baf7a", "bottleneck_only": "#4a3aa7"}
SCHEME_MARK = {"full": "o", "decoder_only": "s", "encoder_only": "^", "bottleneck_only": "D"}
SCHEME_CN = {"full": "Full", "decoder_only": "Decoder-only", "encoder_only": "Encoder-only", "bottleneck_only": "Bottleneck-only"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_BLUE = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
LEVELS = [5, 10, 20, 40]
STARTS = ["id_00", "id_02", "id_04", "id_06", "merged"]
LAYERS = ["encoder.0", "encoder.2", "encoder.4", "decoder.0", "decoder.2", "decoder.4"]
LAYER_CN = {"encoder.0": "enc.0\n320→64", "encoder.2": "enc.2\n64→64", "encoder.4": "enc.4\n64→8",
            "decoder.0": "dec.0\n8→64", "decoder.2": "dec.2\n64→64", "decoder.4": "dec.4\n64→320"}

# ----------------------------------------------------------------------------- data
b0 = pd.read_csv(RES / "Baseline/logs/stage0_source_baseline.csv").set_index("model")
s1s = pd.read_csv(RES / "Stage1/logs/stage1_small_shift.csv")
s1l = pd.read_csv(RES / "Stage1/logs/stage1_large_shift.csv").set_index("model")
c2 = pd.read_csv(RES / "Stage2/logs/stage2_configs.csv")
r2 = pd.read_csv(RES / "Stage2/logs/stage2_runs.csv")
c3 = pd.read_csv(RES / "Stage3/logs/stage3_configs.csv").set_index("data_level")
cmp3 = pd.read_csv(RES / "Stage3/logs/stage3_vs_stage2_comparison.csv").set_index("data_level")
r3 = pd.read_csv(RES / "Stage3/logs/stage3_runs.csv")

sm = c2[c2["shift"] == "small"].copy()
lg = c2[c2["shift"] == "large"].copy()
assert len(sm) == 192 and len(lg) == 80 and len(r2) == 992 and len(r3) == 40

N: dict[str, dict] = {}          # numbers.json: key -> {value, source}
CAPTIONS: dict[str, str] = {}    # one-sentence conclusion per figure


def rec(key, value, source):
    N[key] = {"value": float(value) if not isinstance(value, (int, np.integer)) else int(value), "source": source}
    return value


def savefig(fig, name, caption):
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    CAPTIONS[name] = caption
    print(f"  {p.name:<36} {caption}")


def strip(ax, x, ys, color, jitter=0.06, size=34, seed=0):
    """One column of scatter points (slight jitter) + 2px surface-colored outline."""
    rng = np.random.default_rng(seed)
    xs = x + rng.uniform(-jitter, jitter, len(ys))
    ax.scatter(xs, ys, s=size, color=color, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)


def mean_tick(ax, x, m, label=None, w=0.28, dy=0.0, above=True, fs=9):
    ax.hlines(m, x - w, x + w, color=INK, linewidth=1.8, zorder=4)
    if label:
        ax.annotate(label, (x + w + 0.04, m), va="center", ha="left", fontsize=fs, color=INK)


def nudge(vals, min_gap):
    """Enforce a minimum gap between directly labelled y values to avoid overlap. Returns adjusted y."""
    order = np.argsort(vals)
    out = np.array(vals, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


# ============================================================================ base numbers
for m in STARTS:
    rec(f"stage0_source_auc_{m}", b0.loc[m, "auc"], "Baseline/logs/stage0_source_baseline.csv: auc")
rec("stage1_small_mean_auc", s1s["auc"].mean(), "Stage1/logs/stage1_small_shift.csv: mean(auc)")
rec("stage1_small_mean_drop", s1s["auc_drop"].mean(), "Stage1/logs/stage1_small_shift.csv: mean(auc_drop)")
rec("stage1_small_mean_ratio", s1s["shift_ratio"].mean(), "Stage1/logs/stage1_small_shift.csv: mean(shift_ratio)")
rec("stage1_large_mean_auc", s1l["pooled_auc"].mean(), "Stage1/logs/stage1_large_shift.csv: mean(pooled_auc)")
rec("stage1_large_mean_drop", s1l["auc_drop"].mean(), "Stage1/logs/stage1_large_shift.csv: mean(auc_drop)")
rec("stage1_large_mean_ratio", s1l["shift_ratio"].mean(), "Stage1/logs/stage1_large_shift.csv: mean(shift_ratio)")
rec("stage1_small_n_below_half", int((s1s["auc"] < 0.5).sum()), "Stage1/logs/stage1_small_shift.csv: count(auc<0.5)")
rec("stage1_small_min_auc", s1s["auc"].min(), "Stage1/logs/stage1_small_shift.csv: min(auc)")
_w = s1s.loc[s1s["auc"].idxmin()]
N["stage1_small_min_auc_pair"] = {"value": f"{_w.source_model}→{_w.target_id}", "source": "Stage1/logs/stage1_small_shift.csv: argmin(auc)"}
for m in STARTS:
    rec(f"stage1_large_pooled_auc_{m}", s1l.loc[m, "pooled_auc"], "Stage1/logs/stage1_large_shift.csv: pooled_auc")
    rec(f"stage1_large_drop_{m}", s1l.loc[m, "auc_drop"], "Stage1/logs/stage1_large_shift.csv: auc_drop")
    rec(f"stage1_large_ratio_{m}", s1l.loc[m, "shift_ratio"], "Stage1/logs/stage1_large_shift.csv: shift_ratio")

# ============================================================================ Fig. 1: RQ1, the two metrics move in opposite directions
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
for ax, col, title, ylab, fmt in [
    (axes[0], "auc_drop", "AUC drop (source AUC - target AUC)", "AUC drop", "{:.3f}"),
    (axes[1], "shift_ratio", "Threshold-shift ratio (target / source median normal MSE)", "Shift ratio (×)", "{:.2f}×"),
]:
    strip(ax, 0, s1s[col].values, BLUE, seed=1)
    strip(ax, 1, s1l[col].values, BLUE, seed=2)
    ms, ml = s1s[col].mean(), s1l[col].mean()
    mean_tick(ax, 0, ms, "mean " + fmt.format(ms))
    mean_tick(ax, 1, ml, "mean " + fmt.format(ml))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Small shift\nMIMII ID → ID\n12 pairs", "Large shift\nindustrial → household\n5 models"])
    ax.set_xlim(-0.6, 2.0); ax.set_ylabel(ylab); ax.set_title(title, loc="left")
axes[0].axhline(0, color=AXIS, linewidth=0.8)
axes[1].axhline(1, color=AXIS, linewidth=0.8)
axes[1].annotate("1× = no shift", (1.55, 1), fontsize=8, color=MUTED, va="bottom")
fig.suptitle("Fig. 1 · Two domain shifts: the one with less threshold shift loses more AUC", x=0.02, ha="left", fontsize=12.5, color=INK, y=1.02)
savefig(fig, "fig1_rq1_drop_vs_ratio",
        f"Small shift moves the threshold only {N['stage1_small_mean_ratio']['value']:.2f}× yet drops {N['stage1_small_mean_drop']['value']:.3f} AUC; "
        f"large shift moves it {N['stage1_large_mean_ratio']['value']:.2f}× yet drops only {N['stage1_large_mean_drop']['value']:.3f}.")

# ============================================================================ Fig. 1b: small shift, ranking inversion
d = s1s.sort_values(["target_id", "source_model"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.0, 4.6))
ax.grid(False); ax.grid(True, axis="x", color=GRID, linewidth=0.6)
ypos, labels, ticks_group = [], [], []
y = 0
for tid, grp in d.groupby("target_id", sort=True):
    for _, r in grp.iterrows():
        flipped = r["auc"] < 0.5
        ax.barh(y, r["auc"], height=0.72, color=BLUE if flipped else "#c3c2b7", zorder=3)
        if flipped:
            ax.annotate(f"{r['auc']:.3f}", (r["auc"] + 0.008, y), va="center", ha="left", fontsize=8.5, color=INK)
        ypos.append(y); labels.append(f"{r['source_model']} → {tid}")
        y += 1
    y += 0.6
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.axvline(0.5, color=INK, linewidth=1.0, linestyle="--", zorder=4)
ax.annotate("AUC 0.5 = chance level", (0.5, -1.0), ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xlim(0, 1.0); ax.set_xlabel("Frozen target AUC (Stage 1, no fine-tuning)")
ax.set_title(f"Fig. 1b · Small shift: {N['stage1_small_n_below_half']['value']} of 12 pairs have AUC below 0.5 -- normal/abnormal ranking is inverted", loc="left", fontsize=12.5, pad=14)
ax.legend(handles=[Line2D([], [], marker="s", color=BLUE, linestyle="", markersize=9, label="AUC < 0.5 (ranking inverted)"),
                   Line2D([], [], marker="s", color="#c3c2b7", linestyle="", markersize=9, label="AUC ≥ 0.5")],
          loc="lower right")
savefig(fig, "fig1b_rq1_small_shift_flip",
        f"In {N['stage1_small_n_below_half']['value']} of 12 small-shift pairs the frozen AUC is < 0.5 (lowest {N['stage1_small_min_auc']['value']:.3f}, "
        f"{N['stage1_small_min_auc_pair']['value']}) -- the model ranks abnormal clips as more 'normal' than normal ones, not just less sharply.")

# ============================================================================ Fig. 2: RQ2, data-efficiency curves
sm_curve = sm.pivot_table(index="layer_scheme", columns="data_level", values="target_auc", aggfunc="mean")
lg_curve = lg.pivot_table(index="layer_scheme", columns="data_level", values="pooled_auc", aggfunc="mean")
for sch in SCHEME_ORDER:
    for n in LEVELS:
        rec(f"stage2_small_auc_{sch}_n{n}", sm_curve.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=small]: mean(target_auc) by layer_scheme,data_level")
        rec(f"stage2_large_auc_{sch}_n{n}", lg_curve.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=large]: mean(pooled_auc) by layer_scheme,data_level")
f = lg_curve.loc["full"]
rec("stage2_large_full_delta_5_10", f[10] - f[5], "derived: stage2_large_auc_full_n10 - n5")
rec("stage2_large_full_delta_10_20", f[20] - f[10], "derived: stage2_large_auc_full_n20 - n10")
rec("stage2_large_full_delta_20_40", f[40] - f[20], "derived: stage2_large_auc_full_n40 - n20")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
xs = np.arange(len(LEVELS))
for ax, curve, frozen, title in [
    (axes[0], sm_curve, N["stage1_small_mean_auc"]["value"], "Small shift (MIMII ID to ID, mean of 12 pairs)"),
    (axes[1], lg_curve, N["stage1_large_mean_auc"]["value"], "Large shift (-> household fan, mean of 5 starts, pooled)"),
]:
    ends = []
    for sch in SCHEME_ORDER:
        ys = [curve.loc[sch, n] for n in LEVELS]
        ax.plot(xs, ys, color=SCHEME_COLOR[sch], linewidth=2, marker=SCHEME_MARK[sch], markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=SCHEME_CN[sch], zorder=3)
        ends.append(ys[-1])
    ax.axhline(frozen, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.annotate(f"frozen baseline {frozen:.3f}", (1.5, frozen), va="center", ha="center", fontsize=8.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fcfcfb", edgecolor="none"))
    lab_y = nudge(ends, 0.028)
    for sch, y0, y1 in zip(SCHEME_ORDER, ends, lab_y):
        ax.annotate(SCHEME_CN[sch], (xs[-1] + 0.12, y1), va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([f"{n} clips" for n in LEVELS]); ax.set_xlim(-0.3, 4.1)
    ax.set_xlabel("Target-domain normal clips used for fine-tuning"); ax.set_title(title, loc="left")
axes[0].set_ylabel("Target AUC after fine-tuning"); axes[0].set_ylim(0.3, 1.02)
# Knee annotation (large shift, full)
ax = axes[1]
d1, d2, d3 = (N[f"stage2_large_full_delta_{k}"]["value"] for k in ("5_10", "10_20", "20_40"))
ax.annotate(f"+{d1:.3f}", (0.5, (f[5] + f[10]) / 2 + 0.03), ha="center", fontsize=8.5, color=BLUE)
ax.annotate(f"+{d2:.3f}", (1.5, (f[10] + f[20]) / 2 + 0.03), ha="center", fontsize=8.5, color=BLUE)
ax.annotate(f"+{d3:.3f}", (2.5, (f[20] + f[40]) / 2 + 0.03), ha="center", fontsize=8.5, color=BLUE)
ax.annotate("knee: gains slow after 20 clips", (xs[2], f[20] - 0.06), ha="center", va="top", fontsize=8.5, color=BLUE)
ax.plot([xs[2]], [f[20]], marker="o", markersize=13, markerfacecolor="none", markeredgecolor=BLUE, markeredgewidth=1.2, zorder=4)
axes[0].legend(loc="lower right", ncol=2)
fig.suptitle("Fig. 2 · Data efficiency: target AUC versus amount of fine-tuning data", x=0.02, ha="left", fontsize=12.5, color=INK, y=1.02)
savefig(fig, "fig2_rq2_data_efficiency",
        f"Large shift, full: 5→10 clips +{d1:.3f}, 10→20 clips +{d2:.3f}, 20→40 clips only +{d3:.3f} -- the knee is at 20 clips. "
        f"In the small shift all four curves are already near their plateaus at 5 clips.")

# ============================================================================ Fig. 3: RQ3, layer schemes (n=40)
g_small = sm[sm.data_level == 40].groupby("layer_scheme")["target_auc_gain"].mean()
g_large = lg[lg.data_level == 40].groupby("layer_scheme")["pooled_auc_gain"].mean()
for sch in SCHEME_ORDER:
    rec(f"stage2_small_gain40_{sch}", g_small[sch], "Stage2/logs/stage2_configs.csv[shift=small,n=40]: mean(target_auc_gain)")
    rec(f"stage2_large_gain40_{sch}", g_large[sch], "Stage2/logs/stage2_configs.csv[shift=large,n=40]: mean(pooled_auc_gain)")
rec("stage2_small_decoder_over_full_pct", 100 * g_small["decoder_only"] / g_small["full"], "derived: small gain40 decoder_only / full")
rec("stage2_large_decoder_over_full_pct", 100 * g_large["decoder_only"] / g_large["full"], "derived: large gain40 decoder_only / full")

fig, ax = plt.subplots(figsize=(8.2, 4.3))
w, gap = 0.19, 0.012
for gi, (name, g) in enumerate([("Small shift", g_small), ("Large shift", g_large)]):
    for si, sch in enumerate(SCHEME_ORDER):
        x = gi + (si - 1.5) * (w + gap)
        v = g[sch]
        ax.bar(x, v, width=w, color=SCHEME_COLOR[sch], zorder=3, label=SCHEME_CN[sch] if gi == 0 else None)
        ax.annotate(f"{v:+.3f}", (x, v + (0.008 if v >= 0 else -0.008)), ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.5, color=INK)
ax.axhline(0, color=AXIS, linewidth=0.9, zorder=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Small shift (MIMII ID to ID)", "Large shift (-> household fan)"])
ax.set_ylabel("Target AUC gain (fine-tuned - frozen baseline)"); ax.set_ylim(-0.11, 0.40)
ax.legend(loc="upper right", ncol=2)
ax.set_title("Fig. 3 · Which layers to tune: gain of the four layer schemes at n=40", loc="left", fontsize=12.5, pad=12)
savefig(fig, "fig3_rq3_layer_schemes",
        f"Small shift: Decoder-only reaches {N['stage2_small_decoder_over_full_pct']['value']:.0f}% of Full, Encoder-only is nearly useless ({g_small['encoder_only']:+.3f}); "
        f"large shift: tuning one end alone is negative (Encoder-only {g_large['encoder_only']:+.3f}, Bottleneck-only {g_large['bottleneck_only']:+.3f}).")

# ============================================================================ Fig. 4: gain vs forgetting
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
size_map = {5: 22, 10: 40, 20: 62, 40: 90}
for ax, df, ycol, title in [(axes[0], sm, "target_auc_gain", "Small shift (192 configurations)"),
                            (axes[1], lg, "pooled_auc_gain", "Large shift (80 configurations)")]:
    ax.grid(True, axis="both", color=GRID, linewidth=0.6)
    for sch in SCHEME_ORDER:
        sub = df[df.layer_scheme == sch]
        ax.scatter(sub["source_auc_forget"], sub[ycol], s=sub["data_level"].map(size_map), color=SCHEME_COLOR[sch],
                   marker=SCHEME_MARK[sch], edgecolor="#fcfcfb", linewidth=1.0, alpha=0.9, zorder=3)
    ax.axhline(0, color=AXIS, linewidth=0.9); ax.axvline(0, color=AXIS, linewidth=0.9)
    ax.set_xlabel("Source forgetting (Stage 0 source AUC - fine-tuned source AUC)"); ax.set_title(title, loc="left")
    ax.set_xlim(-0.05, 0.80)
axes[0].set_ylabel("Target gain (fine-tuned - frozen baseline)")
h1 = [Line2D([], [], marker=SCHEME_MARK[s], color=SCHEME_COLOR[s], linestyle="", markersize=8, label=SCHEME_CN[s]) for s in SCHEME_ORDER]
h2 = [Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=np.sqrt(size_map[n]) * 0.9, label=f"{n} clips") for n in LEVELS]
fig.legend(handles=h1 + h2, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.06), columnspacing=1.4, handletextpad=0.5)
fig.suptitle("Fig. 4 · Gain vs forgetting: top-left is better, but there is no free lunch", x=0.02, ha="left", fontsize=12.5, color=INK, y=1.02)
# Numbers
for sch in SCHEME_ORDER:
    rec(f"stage2_small_forget40_{sch}", sm[(sm.data_level == 40) & (sm.layer_scheme == sch)]["source_auc_forget"].mean(),
        "Stage2/logs/stage2_configs.csv[shift=small,n=40]: mean(source_auc_forget)")
    rec(f"stage2_large_forget40_{sch}", lg[(lg.data_level == 40) & (lg.layer_scheme == sch)]["source_auc_forget"].mean(),
        "Stage2/logs/stage2_configs.csv[shift=large,n=40]: mean(source_auc_forget)")
savefig(fig, "fig4_rq3_gain_vs_forget",
        f"Encoder-only barely forgets (large shift {N['stage2_large_forget40_encoder_only']['value']:.3f}) and barely gains; "
        f"Decoder-only forgets as much as Full in the small shift ({N['stage2_small_forget40_decoder_only']['value']:.2f} vs {N['stage2_small_forget40_full']['value']:.2f}) and slightly less in the large shift ({N['stage2_large_forget40_decoder_only']['value']:.2f} vs {N['stage2_large_forget40_full']['value']:.2f}), still the same order of magnitude; "
        f"Bottleneck-only loses on both counts in the large shift: gain {g_large['bottleneck_only']:+.3f}, forgetting {N['stage2_large_forget40_bottleneck_only']['value']:.3f}.")

# ============================================================================ Fig. 5: merged vs single-ID
lf = lg[lg.layer_scheme == "full"].pivot(index="start", columns="data_level", values="pooled_auc")
merged = lf.loc["merged"]; singles = lf.drop("merged")
for n in LEVELS:
    rec(f"stage2_large_full_merged_n{n}", merged[n], "Stage2/logs/stage2_configs.csv[large,full,start=merged]: pooled_auc")
    rec(f"stage2_large_full_singles_mean_n{n}", singles[n].mean(), "Stage2/logs/stage2_configs.csv[large,full,start!=merged]: mean(pooled_auc)")
    rec(f"stage2_large_full_merged_minus_singles_n{n}", merged[n] - singles[n].mean(), "derived: merged - singles_mean")
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.fill_between(xs, singles.min().values, singles.max().values, color="#e1e0d9", alpha=0.8, zorder=1, label="single-ID starts ×4 (range)")
for m in singles.index:
    ax.plot(xs, singles.loc[m].values, color=MUTED, linewidth=1.1, marker="o", markersize=4, zorder=2)
ax.plot(xs, merged.values, color=BLUE, linewidth=2.2, marker="o", markersize=6.5, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
        zorder=4, label="merged (mixed pretraining on four IDs)")
for i, n in enumerate(LEVELS):
    dlt = merged[n] - singles[n].mean()
    ax.annotate("0.000" if abs(dlt) < 5e-4 else f"{dlt:+.3f}", (xs[i], merged[n] + 0.018), ha="center", va="bottom", fontsize=8.5, color=BLUE)
ax.annotate("merged", (xs[-1] + 0.1, merged[40]), va="center", fontsize=9, color=INK)
ax.annotate("single-ID ×4", (xs[-1] + 0.1, singles[40].mean() - 0.03), va="center", fontsize=9, color=INK2)
ax.set_xticks(xs); ax.set_xticklabels([f"{n} clips" for n in LEVELS]); ax.set_xlim(-0.3, 3.9)
ax.set_ylim(0.68, 1.03); ax.set_xlabel("Target-domain normal clips used for fine-tuning"); ax.set_ylabel("Pooled target AUC (full fine-tuning)")
ax.legend(loc="lower right")
ax.set_title("Fig. 5 · Pretraining diversity: merged leads most at 5-10 clips, zero at 40", loc="left", fontsize=12.5, pad=12)
savefig(fig, "fig5_rq3_merged_vs_single",
        "merged minus single-ID mean: " + ", ".join(f"{n} clips {N[f'stage2_large_full_merged_minus_singles_n{n}']['value']:+.3f}" for n in LEVELS) + ".")

# ============================================================================ Fig. 6: value of pretraining
for n in LEVELS:
    rec(f"stage3_nopretrain_auc_n{n}", cmp3.loc[n, "pooled_auc"], "Stage3/logs/stage3_vs_stage2_comparison.csv: pooled_auc")
    rec(f"stage3_pretrained_auc_n{n}", cmp3.loc[n, "pretrained_auc"], "Stage3/logs/stage3_vs_stage2_comparison.csv: pretrained_auc")
    rec(f"stage3_pretrain_value_n{n}", cmp3.loc[n, "auc_diff"], "Stage3/logs/stage3_vs_stage2_comparison.csv: auc_diff")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.12))
ax1.plot(xs, cmp3.loc[LEVELS, "pretrained_auc"], color=BLUE, linewidth=2.2, marker="o", markersize=6.5,
         markeredgecolor="#fcfcfb", markeredgewidth=1.2, label="pretrained (MIMII pretraining + full fine-tuning, mean of 5 starts)", zorder=3)
ax1.plot(xs, cmp3.loc[LEVELS, "pooled_auc"], color=ORANGE, linewidth=2.2, marker="s", markersize=6,
         markeredgecolor="#fcfcfb", markeredgewidth=1.2, label="no pretraining (random init., trained directly)", zorder=3)
ax1.annotate("pretrained", (xs[-1] + 0.1, cmp3.loc[40, "pretrained_auc"]), va="center", fontsize=9, color=INK)
ax1.annotate("no pretraining", (xs[-1] + 0.1, cmp3.loc[40, "pooled_auc"]), va="center", fontsize=9, color=INK)
ax1.set_ylim(0.6, 1.03); ax1.set_ylabel("Pooled target AUC"); ax1.legend(loc="lower right"); ax1.set_xlim(-0.3, 3.9)
ax1.set_title("Fig. 6 · Value of pretraining: largest at 20 clips, smaller at 40 (inverted U)", loc="left", fontsize=12.5, pad=12)
diffs = cmp3.loc[LEVELS, "auc_diff"].values
ax2.bar(xs, diffs, width=0.5, color=BLUE, zorder=3)
for x, v in zip(xs, diffs):
    ax2.annotate(f"+{v:.3f}", (x, v + 0.008), ha="center", va="bottom", fontsize=8.5, color=INK)
ax2.axhline(0, color=AXIS, linewidth=0.9); ax2.set_ylim(0, 0.30); ax2.set_ylabel("Difference\n(with - without)")
ax2.set_xticks(xs); ax2.set_xticklabels([f"{n} clips" for n in LEVELS]); ax2.set_xlabel("Target-domain normal clips used for training")
savefig(fig, "fig6_pretrain_value",
        "AUC gained from pretraining: " + ", ".join(f"{n} clips +{N[f'stage3_pretrain_value_n{n}']['value']:.3f}" for n in LEVELS)
        + " -- largest at 20 clips; with enough data, training from scratch closes most of the gap.")

# ============================================================================ Fig. 7: weight-drift heatmap
dcols = [f"drift_{l}" for l in LAYERS]
r2s = r2[(r2["shift"] == "small") & (r2.data_level == 40)]
r2l = r2[(r2["shift"] == "large") & (r2.data_level == 40)]
H_small = r2s.groupby("layer_scheme")[dcols].mean().reindex(SCHEME_ORDER)
H_large = r2l.groupby("layer_scheme")[dcols].mean().reindex(SCHEME_ORDER)
for sch in SCHEME_ORDER:
    for l, c in zip(LAYERS, dcols):
        rec(f"drift_large40_{sch}_{l}", H_large.loc[sch, c], "Stage2/logs/stage2_runs.csv[large,n=40]: mean(drift_<layer>) by layer_scheme")
        rec(f"drift_small40_{sch}_{l}", H_small.loc[sch, c], "Stage2/logs/stage2_runs.csv[small,n=40]: mean(drift_<layer>) by layer_scheme")
vmax = max(H_small.values.max(), H_large.values.max())
fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), gridspec_kw=dict(width_ratios=[1, 1.08]))
for ax, H, title in [(axes[0], H_small, "Small shift (n=40, mean of 192 runs)"), (axes[1], H_large, "Large shift (n=40, mean of 800 runs)")]:
    ax.grid(False)
    im = ax.imshow(H.values, cmap=CMAP_BLUE, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels([LAYER_CN[l] for l in LAYERS], fontsize=8.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([SCHEME_CN[s] for s in SCHEME_ORDER], fontsize=9.5)
    for i in range(4):
        for j in range(6):
            v = H.values[i, j]
            ax.text(j, i, "frozen" if v == 0 else f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color=("#fcfcfb" if v > 0.55 * vmax else INK) if v > 0 else MUTED)
    ax.set_title(title, loc="left", pad=26)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.axvline(2.5, color="#fcfcfb", linewidth=3)
    ax.annotate("Encoder", (1, -0.85), ha="center", fontsize=9, color=INK2, annotation_clip=False)
    ax.annotate("Decoder", (4, -0.85), ha="center", fontsize=9, color=INK2, annotation_clip=False)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("Relative weight drift ‖ΔW‖ / ‖W‖", fontsize=9); cb.outline.set_visible(False)
fig.suptitle("Fig. 7 · Which layers each scheme actually moves (relative weight change before vs after fine-tuning)", x=0.02, ha="left", fontsize=12.5, color=INK, y=1.12)
savefig(fig, "fig7_rq3_weight_drift",
        f"Under full fine-tuning the output layer dec.4 moves most (large shift {N['drift_large40_full_decoder.4']['value']:.2f}) -- adaptation is mainly the decoder's job; "
        f"Bottleneck-only moves enc.4 by {N['drift_large40_bottleneck_only_encoder.4']['value']:.2f} (largest in the table) with no downstream layer to absorb it, hence the negative gain.")

# ============================================================================ Fig. 8: how "perfect" are the results
fold40 = r2[(r2["shift"] == "large") & (r2.data_level == 40) & (r2.layer_scheme == "full")]
n_perfect = int((fold40["target_auc"] == 1.0).sum())
rec("stage2_large_full40_folds_auc_eq_1", n_perfect, "Stage2/logs/stage2_runs.csv[large,n=40,full]: count(target_auc==1.0)")
rec("stage2_large_full40_folds_total", len(fold40), "Stage2/logs/stage2_runs.csv[large,n=40,full]: count")
rec("stage2_large_full40_fold_auc_min", fold40["target_auc"].min(), "Stage2/logs/stage2_runs.csv[large,n=40,full]: min(target_auc)")
for n in LEVELS:
    rec(f"stage3_fold_auc_mean_n{n}", c3.loc[n, "fold_auc_mean"], "Stage3/logs/stage3_configs.csv: fold_auc_mean")
    rec(f"stage3_fold_auc_std_n{n}", c3.loc[n, "fold_auc_std"], "Stage3/logs/stage3_configs.csv: fold_auc_std")
rec("stage3_fold_auc_min_n5", r3[r3.data_level == 5]["target_auc"].min(), "Stage3/logs/stage3_runs.csv[n=5]: min(target_auc)")
rec("stage3_fold_auc_max_n40", r3[r3.data_level == 40]["target_auc"].max(), "Stage3/logs/stage3_runs.csv[n=40]: max(target_auc)")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
for i, m in enumerate(STARTS):
    ys = fold40[fold40.start == m]["target_auc"].values
    strip(ax, i, ys, BLUE, jitter=0.12, size=30, seed=i + 10)
ax.set_xticks(range(5)); ax.set_xticklabels(STARTS); ax.set_xlabel("Pretraining start")
ax.set_title("Pretrained + fine-tuned (n=40, full): per-fold AUC", loc="left")
ax.annotate(f"{n_perfect}/{len(fold40)} folds with AUC exactly = 1.000", (2, 0.975), ha="center", va="top", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fcfcfb", edgecolor=AXIS, linewidth=0.8))
ax.set_ylabel("Per-fold AUC (6 normal + 12 abnormal per fold)")
ax = axes[1]
for i, n in enumerate(LEVELS):
    ys = r3[r3.data_level == n]["target_auc"].values
    strip(ax, i, ys, ORANGE, jitter=0.12, size=30, seed=i + 20)
    mean_tick(ax, i, ys.mean(), f"{ys.mean():.3f}", w=0.25)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{n} clips" for n in LEVELS]); ax.set_xlabel("Normal clips used for training")
ax.set_title("No pretraining (random init.): per-fold AUC", loc="left")
ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle="--")
ax.annotate("chance 0.5", (3.45, 0.5), va="bottom", ha="right", fontsize=8.5, color=INK2)
axes[0].set_ylim(0.42, 1.03)
fig.suptitle("Fig. 8 · Limitation: these recordings are easy to separate", x=0.02, ha="left", fontsize=12.5, color=INK, y=1.02)
savefig(fig, "fig8_limitation_too_perfect",
        f"{n_perfect}/{len(fold40)} folds have AUC exactly 1.0; random init. with only 5 clips already reaches a fold mean of {N['stage3_fold_auc_mean_n5']['value']:.3f} (pooled {N['stage3_nopretrain_auc_n5']['value']:.3f}), "
        f"and {N['stage3_fold_auc_mean_n40']['value']:.3f} at 40 clips (pooled {N['stage3_nopretrain_auc_n40']['value']:.3f}) -- the 10 rounds recorded on the same day with the same device may still be too similar.")

# ============================================================================ other numbers used in the report
# Stronger source -> more forgetting (small shift, n=40, full, by start)
per_start = sm[(sm.data_level == 40) & (sm.layer_scheme == "full")].groupby("start")[["source_auc_forget", "target_auc_gain"]].mean()
per_target = sm[(sm.data_level == 40) & (sm.layer_scheme == "full")].groupby("target")[["frozen_target_auc", "target_auc", "target_auc_gain"]].mean()
for m in ["id_00", "id_02", "id_04", "id_06"]:
    rec(f"stage2_small_full40_forget_by_start_{m}", per_start.loc[m, "source_auc_forget"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(source_auc_forget) by start")
    rec(f"stage2_small_full40_frozen_by_target_{m}", per_target.loc[m, "frozen_target_auc"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(frozen_target_auc) by target")
    rec(f"stage2_small_full40_auc_by_target_{m}", per_target.loc[m, "target_auc"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(target_auc) by target")
    rec(f"stage2_small_full40_gain_by_target_{m}", per_target.loc[m, "target_auc_gain"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(target_auc_gain) by target")
lps = lg[(lg.data_level == 40) & (lg.layer_scheme == "full")].set_index("start")
for m in STARTS:
    rec(f"stage2_large_full40_forget_{m}", lps.loc[m, "source_auc_forget"], "Stage2/logs/stage2_configs.csv[large,n=40,full]: source_auc_forget")
    rec(f"stage2_large_full40_pooled_auc_{m}", lps.loc[m, "pooled_auc"], "Stage2/logs/stage2_configs.csv[large,n=40,full]: pooled_auc")
    rec(f"stage2_large_full40_fold_std_{m}", lps.loc[m, "fold_auc_std"], "Stage2/logs/stage2_configs.csv[large,n=40,full]: fold_auc_std")
rec("total_training_runs", 5 + 992 + 40, "derived: stage0 5 + stage2 992 + stage3 40")
rec("stage2_runs", len(r2), "Stage2/logs/stage2_runs.csv: count"); rec("stage2_configs", len(c2), "Stage2/logs/stage2_configs.csv: count")
rec("stage2_failures", 0, "Stage2/logs/stage2_errors.jsonl: absent")

# ============================================================================ write to disk
with open(OUT / "numbers.json", "w", encoding="utf-8") as fh:
    json.dump(N, fh, ensure_ascii=False, indent=1)
with open(OUT / "captions.md", "w", encoding="utf-8") as fh:
    for k, v in CAPTIONS.items():
        fh.write(f"- **{k}**: {v}\n")


def md_table(df, fmt="{:.4f}", index_name=""):
    cols = list(df.columns)
    lines = ["| " + index_name + " | " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


T = []
T.append("### Table A1 · Stage 0 source baseline and Stage 1 frozen evaluation (large shift)\n")
t = s1l[["source_auc", "pooled_auc", "auc_drop", "fold_auc_std", "shift_ratio"]].loc[STARTS].copy()
t.columns = ["Source AUC (Stage 0)", "Large-shift pooled AUC (frozen)", "AUC drop", "Fold std", "Shift ratio"]
T.append(md_table(t, index_name="Model") + "\n")
T.append("### Table A2 · Stage 1 small shift, 12 pairs (frozen evaluation)\n")
t = s1s.set_index(s1s["source_model"] + " → " + s1s["target_id"])[["source_auc", "auc", "auc_drop", "shift_ratio"]]
t.columns = ["Source AUC", "Target AUC", "AUC drop", "Shift ratio"]
T.append(md_table(t, index_name="Start → target") + "\n")
T.append("### Table A3 · Stage 2 target AUC (scheme × data level)\n\nSmall shift (mean of 12 pairs, target_auc):\n")
t = sm_curve.loc[SCHEME_ORDER]; t.index = [SCHEME_CN[s] for s in t.index]; t.columns = [f"{n} clips" for n in t.columns]
T.append(md_table(t, index_name="Layer scheme") + "\n\nLarge shift (mean of 5 starts, pooled_auc):\n")
t = lg_curve.loc[SCHEME_ORDER]; t.index = [SCHEME_CN[s] for s in t.index]; t.columns = [f"{n} clips" for n in t.columns]
T.append(md_table(t, index_name="Layer scheme") + "\n")
T.append("### Table A4 · Stage 2 gain and forgetting (scheme × data level)\n\nSmall-shift gain (target_auc_gain):\n")
t = sm.pivot_table(index="layer_scheme", columns="data_level", values="target_auc_gain", aggfunc="mean").loc[SCHEME_ORDER]
t.index = [SCHEME_CN[s] for s in t.index]; t.columns = [f"{n} clips" for n in t.columns]
T.append(md_table(t, "{:+.4f}", "Layer scheme") + "\n\nLarge-shift gain (pooled_auc_gain):\n")
t = lg.pivot_table(index="layer_scheme", columns="data_level", values="pooled_auc_gain", aggfunc="mean").loc[SCHEME_ORDER]
t.index = [SCHEME_CN[s] for s in t.index]; t.columns = [f"{n} clips" for n in t.columns]
T.append(md_table(t, "{:+.4f}", "Layer scheme") + "\n\nSmall-shift forgetting (source_auc_forget):\n")
t = sm.pivot_table(index="layer_scheme", columns="data_level", values="source_auc_forget", aggfunc="mean").loc[SCHEME_ORDER]
t.index = [SCHEME_CN[s] for s in t.index]; t.columns = [f"{n} clips" for n in t.columns]
T.append(md_table(t, "{:+.4f}", "Layer scheme") + "\n\nLarge-shift forgetting (source_auc_forget):\n")
t = lg.pivot_table(index="layer_scheme", columns="data_level", values="source_auc_forget", aggfunc="mean").loc[SCHEME_ORDER]
t.index = [SCHEME_CN[s] for s in t.index]; t.columns = [f"{n} clips" for n in t.columns]
T.append(md_table(t, "{:+.4f}", "Layer scheme") + "\n")
T.append("### Table A5 · Relative weight drift ‖ΔW‖/‖W‖ (n=40, large shift)\n")
t = H_large.copy(); t.index = [SCHEME_CN[s] for s in t.index]; t.columns = LAYERS
T.append(md_table(t, "{:.3f}", "Layer scheme") + "\n")
T.append("### Table A6 · With / without pretraining (full, pooled AUC)\n")
t = cmp3.loc[LEVELS, ["pooled_auc", "pretrained_auc", "auc_diff", "fold_auc_std"]].copy()
t.index = [f"{n} clips" for n in t.index]; t.columns = ["No-pretraining AUC", "Pretrained AUC (Stage 2 full, mean of 5 starts)", "Difference", "No-pretraining fold std"]
T.append(md_table(t, index_name="Data level") + "\n")
T.append("### Table A7 · merged vs single-ID (large shift, full, pooled AUC)\n")
t = lf.loc[STARTS].copy(); t.columns = [f"{n} clips" for n in t.columns]
T.append(md_table(t, index_name="Start") + "\n")
T.append("### Table A8 · Large-shift full fold AUC std (start × data level, 10 folds)\n")
t = lg[lg.layer_scheme == "full"].pivot(index="start", columns="data_level", values="fold_auc_std").loc[STARTS]
for m in STARTS:
    for n in LEVELS:
        rec(f"stage2_large_full_fold_std_{m}_n{n}", t.loc[m, n], "Stage2/logs/stage2_configs.csv[large,full]: fold_auc_std")
t.columns = [f"{n} clips" for n in t.columns]
T.append(md_table(t, "{:.4f}", "Start") + "\n")
with open(OUT / "numbers.json", "w", encoding="utf-8") as fh:      # rewrite: includes the A8 keys
    json.dump(N, fh, ensure_ascii=False, indent=1)
with open(OUT / "tables.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(T))

print(f"\n✅ {len(CAPTIONS)} figures, numbers.json with {len(N)} numbers, tables.md, captions.md → {OUT}")
print(f"   font: {FONT}")
