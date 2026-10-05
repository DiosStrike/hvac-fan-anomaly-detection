# -*- coding: utf-8 -*-
"""Figures, tables and number registry for the HVAC 2.0 report -- reads only raw data in results/, no dependence on older figures.

Outputs
  report_figures/original/figN_*.{pdf,png}   titled version (suptitle on top)
  report_figures/clean/figN_*.{pdf,png}      untitled version (the LaTeX caption carries the description)
  report/figures/figN_*.pdf                  copy of the clean PDFs, referenced directly by LaTeX
  report/tables/*.tex                        data-generated LaTeX table fragments (\\input; since round 3 only the 4 still used by the report)
  report/numbers_audit.json                  every number cited in the report -> source file / column / computation

Usage: python code/analysis/make_report_figures.py   (from the repo root)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
ORIG = ROOT / "report_figures" / "original"
CLEAN = ROOT / "report_figures" / "clean"
REP_FIG = ROOT / "report" / "figures"
REP_TAB = ROOT / "report" / "tables"
for d in (ORIG, CLEAN, REP_FIG, REP_TAB):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- style
_FONTS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]   # round 5: English version, single sans-serif font
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _FONTS if f in _avail), None)
plt.rcParams.update({
    "font.family": [FONT] if FONT else ["sans-serif"],
    "axes.unicode_minus": False, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": "#6b6a66", "ytick.color": "#6b6a66", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelcolor": "#3d3c39", "axes.labelsize": 8.5,
    "axes.titlecolor": "#0b0b0b", "axes.titlesize": 9, "axes.titleweight": "medium",
    "legend.frameon": False, "legend.fontsize": 8, "legend.handlelength": 1.6,
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Palette (checked with the dataviz validator): fixed order for the four layer schemes; merged is always a red diamond
SCHEME_ORDER = ["full", "decoder_only", "encoder_only", "bottleneck_only"]
SCHEME_COLOR = {"full": "#2a78d6", "decoder_only": "#eb6834", "encoder_only": "#1baf7a", "bottleneck_only": "#4a3aa7"}
SCHEME_MARK = {"full": "o", "decoder_only": "s", "encoder_only": "^", "bottleneck_only": "D"}
SCHEME_CN = {"full": "Full", "decoder_only": "Decoder-only", "encoder_only": "Encoder-only", "bottleneck_only": "Bottleneck-only"}   # variable name kept, labels are English (see report/GLOSSARY.md)
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
MERGED_COLOR, MERGED_MARK = RED, "D"
INK, INK2, MUTED, GRID, AXIS, BASE_GREY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#c9c8c1"
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_BLUE = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
LEVELS = [5, 10, 20, 40]
IDS = ["id_00", "id_02", "id_04", "id_06"]
STARTS = IDS + ["merged"]
ID_MARK = {"id_00": "o", "id_02": "s", "id_04": "^", "id_06": "v", "merged": MERGED_MARK}
LAYERS = ["encoder.0", "encoder.2", "encoder.4", "decoder.0", "decoder.2", "decoder.4"]
LAYER_CN = {"encoder.0": "enc.0\n320-64", "encoder.2": "enc.2\n64-64", "encoder.4": "enc.4\n64-8",
            "decoder.0": "dec.0\n8-64", "decoder.2": "dec.2\n64-64", "decoder.4": "dec.4\n64-320"}
NAME_CN = {"id_00": "id_00", "id_02": "id_02", "id_04": "id_04", "id_06": "id_06", "merged": "merged"}

# ----------------------------------------------------------------------------- data (raw files only)
b0 = pd.read_csv(RES / "Baseline/logs/stage0_source_baseline.csv").set_index("model")
s1s = pd.read_csv(RES / "Stage1/logs/stage1_small_shift.csv")
s1l = pd.read_csv(RES / "Stage1/logs/stage1_large_shift.csv").set_index("model")
s1f = pd.read_csv(RES / "Stage1/logs/stage1_large_shift_folds.csv")
c2 = pd.read_csv(RES / "Stage2/logs/stage2_configs.csv")
r2 = pd.read_csv(RES / "Stage2/logs/stage2_runs.csv")
c3 = pd.read_csv(RES / "Stage3/logs/stage3_configs.csv").set_index("data_level")
r3 = pd.read_csv(RES / "Stage3/logs/stage3_runs.csv")
splits = json.load(open(RES / "Baseline/logs/stage0_splits_seed42.json"))
sm = c2[c2["shift"] == "small"].copy()
lg = c2[c2["shift"] == "large"].copy()
assert len(sm) == 192 and len(lg) == 80 and len(r2) == 992 and len(r3) == 40 and len(s1s) == 12 and len(s1f) == 50

N: dict[str, dict] = {}


def rec(key, value, source):
    if isinstance(value, (np.integer, int)):
        v = int(value)
    elif isinstance(value, (float, np.floating)):
        v = float(value)
    else:
        v = value
    N[key] = {"value": v, "source": source}
    return value


def save_both(build, name):
    """build(with_title) -> fig. Writes original (titled) and clean (untitled) versions."""
    for with_title, folder in ((True, ORIG), (False, CLEAN)):
        fig = build(with_title)
        for ext in ("pdf", "png"):
            fig.savefig(folder / f"{name}.{ext}")
        plt.close(fig)
    shutil.copy(CLEAN / f"{name}.pdf", REP_FIG / f"{name}.pdf")
    print(f"  {name}")


# Since round 3: figures default to single-column width (acmart sigconf \columnwidth ~ 3.33 in), inserted at 1:1 so font sizes are final
COLW = 2.95        # round 4: single-column figure width ~88% of \columnwidth (3.35 in); LaTeX inserts at original size, font size unchanged
S_PT = 30                      # scatter s (marker area, pt^2): same for all points of a kind throughout
MS_PT = S_PT ** 0.5            # Line2D markersize (diameter, pt): visually matches scatter s, used in legends


def legend_above(fig, handles, ncol, y=1.0, **kw):
    """Legends always sit outside the plot area (above the figure); savefig(bbox='tight') includes them."""
    opts = dict(handletextpad=0.4, columnspacing=0.9, borderaxespad=0.0); opts.update(kw)
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, **opts)


def suptitle(fig, with_title, text, y=None):
    """Title for the original version: placed above everything (including external legends); not drawn for clean."""
    if not with_title:
        return
    fig.canvas.draw()
    top = 1.0
    for lg in fig.legends:
        bb = lg.get_window_extent().transformed(fig.transFigure.inverted())
        top = max(top, bb.y1)
    fig.suptitle(text, x=0.0, ha="left", fontsize=10, color=INK, y=(y if y is not None else top + 0.025))


def strip(ax, x, ys, color, marker="o", jitter=0.06, size=S_PT, seed=0, **kw):
    rng = np.random.default_rng(seed)
    xs = x + rng.uniform(-jitter, jitter, len(ys))
    ax.scatter(xs, ys, s=size, color=color, marker=marker, edgecolor="white", linewidth=1.1, zorder=3, **kw)


def mean_tick(ax, x, m, label=None, w=0.28, fs=8.5):
    ax.hlines(m, x - w, x + w, color=INK, linewidth=1.7, zorder=4)
    if label:
        ax.annotate(label, (x + w + 0.04, m), va="center", ha="left", fontsize=fs, color=INK)


def nudge(vals, min_gap):
    order = np.argsort(vals)
    out = np.array(vals, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


# ============================================================================ base number registry
for m in STARTS:
    rec(f"stage0_source_auc_{m}", b0.loc[m, "auc"], "Baseline/logs/stage0_source_baseline.csv: auc")
    rec(f"stage0_source_f1_{m}", b0.loc[m, "best_f1"], "Baseline/logs/stage0_source_baseline.csv: best_f1")
    rec(f"stage0_n_train_clips_{m}", b0.loc[m, "n_train_clips"], "Baseline/logs/stage0_source_baseline.csv: n_train_clips")
for m in STARTS:
    sp = splits["splits"][m]
    rec(f"split_train_pool_{m}", len(sp["train_pool"]), "Baseline/logs/stage0_splits_seed42.json: len(train_pool)")
    rec(f"split_eval_normal_{m}", len(sp["eval_normal"]), "Baseline/logs/stage0_splits_seed42.json: len(eval_normal)")
    rec(f"split_eval_abnormal_{m}", len(sp["eval_abnormal"]), "Baseline/logs/stage0_splits_seed42.json: len(eval_abnormal)")

rec("stage1_small_mean_auc", s1s["auc"].mean(), "Stage1/logs/stage1_small_shift.csv: mean(auc)")
rec("stage1_small_mean_drop", s1s["auc_drop"].mean(), "Stage1/logs/stage1_small_shift.csv: mean(auc_drop)")
rec("stage1_small_mean_ratio", s1s["shift_ratio"].mean(), "Stage1/logs/stage1_small_shift.csv: mean(shift_ratio)")
rec("stage1_small_min_drop", s1s["auc_drop"].min(), "Stage1/logs/stage1_small_shift.csv: min(auc_drop)")
rec("stage1_small_max_drop", s1s["auc_drop"].max(), "Stage1/logs/stage1_small_shift.csv: max(auc_drop)")
rec("stage1_small_min_ratio", s1s["shift_ratio"].min(), "Stage1/logs/stage1_small_shift.csv: min(shift_ratio)")
rec("stage1_small_max_ratio", s1s["shift_ratio"].max(), "Stage1/logs/stage1_small_shift.csv: max(shift_ratio)")
rec("stage1_small_n_below_half", int((s1s["auc"] < 0.5).sum()), "Stage1/logs/stage1_small_shift.csv: count(auc<0.5)")
rec("stage1_small_min_auc", s1s["auc"].min(), "Stage1/logs/stage1_small_shift.csv: min(auc)")
_w = s1s.loc[s1s["auc"].idxmin()]
N["stage1_small_min_auc_pair"] = {"value": f"{_w.source_model}->{_w.target_id}", "source": "Stage1/logs/stage1_small_shift.csv: argmin(auc)"}
rec("stage1_small_max_auc", s1s["auc"].max(), "Stage1/logs/stage1_small_shift.csv: max(auc)")
for r in s1s.itertuples():
    k = f"{r.source_model}_to_{r.target_id}"
    rec(f"stage1_small_auc_{k}", r.auc, "Stage1/logs/stage1_small_shift.csv: auc")
    rec(f"stage1_small_drop_{k}", r.auc_drop, "Stage1/logs/stage1_small_shift.csv: auc_drop")
    rec(f"stage1_small_ratio_{k}", r.shift_ratio, "Stage1/logs/stage1_small_shift.csv: shift_ratio")
by_t = s1s.groupby("target_id")["auc"].mean()
by_s = s1s.groupby("source_model")["auc_drop"].mean()
for m in IDS:
    rec(f"stage1_small_mean_auc_by_target_{m}", by_t[m], "Stage1/logs/stage1_small_shift.csv: mean(auc) by target_id")
    rec(f"stage1_small_mean_drop_by_source_{m}", by_s[m], "Stage1/logs/stage1_small_shift.csv: mean(auc_drop) by source_model")

rec("stage1_large_mean_auc", s1l["pooled_auc"].mean(), "Stage1/logs/stage1_large_shift.csv: mean(pooled_auc)")
rec("stage1_large_mean_drop", s1l["auc_drop"].mean(), "Stage1/logs/stage1_large_shift.csv: mean(auc_drop)")
rec("stage1_large_mean_ratio", s1l["shift_ratio"].mean(), "Stage1/logs/stage1_large_shift.csv: mean(shift_ratio)")
rec("stage1_large_min_auc", s1l["pooled_auc"].min(), "Stage1/logs/stage1_large_shift.csv: min(pooled_auc)")
rec("stage1_large_max_auc", s1l["pooled_auc"].max(), "Stage1/logs/stage1_large_shift.csv: max(pooled_auc)")
rec("stage1_large_min_ratio", s1l["shift_ratio"].min(), "Stage1/logs/stage1_large_shift.csv: min(shift_ratio)")
rec("stage1_large_max_ratio", s1l["shift_ratio"].max(), "Stage1/logs/stage1_large_shift.csv: max(shift_ratio)")
for m in STARTS:
    rec(f"stage1_large_pooled_auc_{m}", s1l.loc[m, "pooled_auc"], "Stage1/logs/stage1_large_shift.csv: pooled_auc")
    rec(f"stage1_large_drop_{m}", s1l.loc[m, "auc_drop"], "Stage1/logs/stage1_large_shift.csv: auc_drop")
    rec(f"stage1_large_ratio_{m}", s1l.loc[m, "shift_ratio"], "Stage1/logs/stage1_large_shift.csv: shift_ratio")
    rec(f"stage1_large_fold_auc_mean_{m}", s1l.loc[m, "fold_auc_mean"], "Stage1/logs/stage1_large_shift.csv: fold_auc_mean")
    rec(f"stage1_large_fold_auc_std_{m}", s1l.loc[m, "fold_auc_std"], "Stage1/logs/stage1_large_shift.csv: fold_auc_std")
    rec(f"stage1_large_src_mse_median_{m}", s1l.loc[m, "source_normal_mse_median"], "Stage1/logs/stage1_large_shift.csv: source_normal_mse_median")
    rec(f"stage1_large_tgt_mse_median_{m}", s1l.loc[m, "target_normal_mse_median"], "Stage1/logs/stage1_large_shift.csv: target_normal_mse_median")

# Stage 2 curves and gain/forgetting
sm_curve = sm.pivot_table(index="layer_scheme", columns="data_level", values="target_auc", aggfunc="mean")
lg_curve = lg.pivot_table(index="layer_scheme", columns="data_level", values="pooled_auc", aggfunc="mean")
sm_gain = sm.pivot_table(index="layer_scheme", columns="data_level", values="target_auc_gain", aggfunc="mean")
lg_gain = lg.pivot_table(index="layer_scheme", columns="data_level", values="pooled_auc_gain", aggfunc="mean")
sm_forget = sm.pivot_table(index="layer_scheme", columns="data_level", values="source_auc_forget", aggfunc="mean")
lg_forget = lg.pivot_table(index="layer_scheme", columns="data_level", values="source_auc_forget", aggfunc="mean")
lg_fstd = lg.pivot_table(index="layer_scheme", columns="data_level", values="fold_auc_std", aggfunc="mean")
for sch in SCHEME_ORDER:
    for n in LEVELS:
        rec(f"stage2_small_auc_{sch}_n{n}", sm_curve.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=small]: mean(target_auc) by layer_scheme,data_level")
        rec(f"stage2_large_auc_{sch}_n{n}", lg_curve.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=large]: mean(pooled_auc) by layer_scheme,data_level")
        rec(f"stage2_small_gain_{sch}_n{n}", sm_gain.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=small]: mean(target_auc_gain)")
        rec(f"stage2_large_gain_{sch}_n{n}", lg_gain.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=large]: mean(pooled_auc_gain)")
        rec(f"stage2_small_forget_{sch}_n{n}", sm_forget.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=small]: mean(source_auc_forget)")
        rec(f"stage2_large_forget_{sch}_n{n}", lg_forget.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=large]: mean(source_auc_forget)")
        rec(f"stage2_large_fold_std_{sch}_n{n}", lg_fstd.loc[sch, n], "Stage2/logs/stage2_configs.csv[shift=large]: mean(fold_auc_std)")
f = lg_curve.loc["full"]
rec("stage2_large_full_delta_5_10", f[10] - f[5], "derived: stage2_large_auc_full_n10 - n5")
rec("stage2_large_full_delta_10_20", f[20] - f[10], "derived: stage2_large_auc_full_n20 - n10")
rec("stage2_large_full_delta_20_40", f[40] - f[20], "derived: stage2_large_auc_full_n40 - n20")
fs = sm_curve.loc["full"]
rec("stage2_small_full_delta_5_40", fs[40] - fs[5], "derived: stage2_small_auc_full_n40 - n5")
rec("stage2_small_decoder_over_full_pct_n40", 100 * sm_gain.loc["decoder_only", 40] / sm_gain.loc["full", 40], "derived: small gain n40 decoder_only/full ×100")
rec("stage2_large_decoder_over_full_pct_n40", 100 * lg_gain.loc["decoder_only", 40] / lg_gain.loc["full", 40], "derived: large gain n40 decoder_only/full ×100")
for n in LEVELS:
    rec(f"stage2_small_decoder_over_full_pct_n{n}", 100 * sm_gain.loc["decoder_only", n] / sm_gain.loc["full", n], "derived: small gain decoder_only/full ×100")
    rec(f"stage2_large_decoder_over_full_pct_n{n}", 100 * lg_gain.loc["decoder_only", n] / lg_gain.loc["full", n], "derived: large gain decoder_only/full ×100")
rec("stage2_configs", len(c2), "Stage2/logs/stage2_configs.csv: rows")
rec("stage2_runs", len(r2), "Stage2/logs/stage2_runs.csv: rows")
rec("stage2_runs_small", int((r2["shift"] == "small").sum()), "Stage2/logs/stage2_runs.csv: count(shift=small)")
rec("stage2_runs_large", int((r2["shift"] == "large").sum()), "Stage2/logs/stage2_runs.csv: count(shift=large)")
rec("stage2_failures", 0, "Stage2/logs/stage2_errors.jsonl: file absent")
for sch in SCHEME_ORDER:
    rec(f"params_trainable_{sch}", int(r2[r2.layer_scheme == sch]["n_trainable_params"].iloc[0]), "Stage2/logs/stage2_runs.csv: n_trainable_params (unique per layer_scheme)")
for n in LEVELS:
    sub = r2[r2.data_level == n]
    rec(f"stage2_n_train_vectors_n{n}", int(sub["n_train_vectors"].iloc[0]), "Stage2/logs/stage2_runs.csv: n_train_vectors (unique per data_level)")
    rec(f"stage2_steps_n{n}", int(sub["steps"].iloc[0]), "Stage2/logs/stage2_runs.csv: steps (unique per data_level)")
rec("vectors_per_clip", int(r2[r2.data_level == 5]["n_train_vectors"].iloc[0] // 5), "derived: n_train_vectors(n=5)/5")

# Top-left region (gain vs forgetting)
FORGET_T, GAIN_T = 0.15, 0.30
rec("topleft_forget_threshold", FORGET_T, "definition (threshold in text)")
rec("topleft_gain_threshold", GAIN_T, "definition (threshold in text)")
tl_small = sm[(sm.source_auc_forget < FORGET_T) & (sm.target_auc_gain > GAIN_T)]
tl_large = lg[(lg.source_auc_forget < FORGET_T) & (lg.pooled_auc_gain > GAIN_T)]
rec("topleft_small_count", len(tl_small), f"Stage2/logs/stage2_configs.csv[small]: count(source_auc_forget<{FORGET_T} & target_auc_gain>{GAIN_T})")
rec("topleft_large_count", len(tl_large), f"Stage2/logs/stage2_configs.csv[large]: count(source_auc_forget<{FORGET_T} & pooled_auc_gain>{GAIN_T})")
for _sch in SCHEME_ORDER:
    rec(f"topleft_small_count_{_sch}", int((tl_small.layer_scheme == _sch).sum()), "as above: count by layer_scheme")
for _n in LEVELS:
    rec(f"topleft_small_count_n{_n}", int((tl_small.data_level == _n).sum()), "as above: count by data_level")
N["topleft_small_members"] = {"value": sorted({f"{r.start}->{r.target}" for r in tl_small.itertuples()}), "source": "as above: unique start->target"}
N["topleft_small_schemes"] = {"value": tl_small.groupby("layer_scheme").size().to_dict(), "source": "as above: count by layer_scheme"}
N["topleft_small_levels"] = {"value": tl_small.groupby("data_level").size().to_dict(), "source": "as above: count by data_level"}
rec("topleft_small_max_forget", tl_small["source_auc_forget"].max() if len(tl_small) else float("nan"), "as above: max(source_auc_forget)")
rec("topleft_small_min_gain", tl_small["target_auc_gain"].min() if len(tl_small) else float("nan"), "as above: min(target_auc_gain)")
rec("stage0_source_auc_id_00_ref", b0.loc["id_00", "auc"], "Baseline/logs/stage0_source_baseline.csv: auc[id_00] (used to explain the top-left cluster)")

# Grouping by start/target (small shift, n=40, full)
ps = sm[(sm.data_level == 40) & (sm.layer_scheme == "full")].groupby("start")[["source_auc_forget", "target_auc_gain"]].mean()
pt = sm[(sm.data_level == 40) & (sm.layer_scheme == "full")].groupby("target")[["frozen_target_auc", "target_auc", "target_auc_gain"]].mean()
for m in IDS:
    rec(f"stage2_small_full40_forget_by_start_{m}", ps.loc[m, "source_auc_forget"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(source_auc_forget) by start")
    rec(f"stage2_small_full40_gain_by_start_{m}", ps.loc[m, "target_auc_gain"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(target_auc_gain) by start")
    rec(f"stage2_small_full40_frozen_by_target_{m}", pt.loc[m, "frozen_target_auc"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(frozen_target_auc) by target")
    rec(f"stage2_small_full40_auc_by_target_{m}", pt.loc[m, "target_auc"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(target_auc) by target")
    rec(f"stage2_small_full40_gain_by_target_{m}", pt.loc[m, "target_auc_gain"], "Stage2/logs/stage2_configs.csv[small,n=40,full]: mean(target_auc_gain) by target")

# Weight drift
dcols = [f"drift_{l}" for l in LAYERS]
H_small = r2[(r2["shift"] == "small") & (r2.data_level == 40)].groupby("layer_scheme")[dcols].mean().reindex(SCHEME_ORDER)
H_large = r2[(r2["shift"] == "large") & (r2.data_level == 40)].groupby("layer_scheme")[dcols].mean().reindex(SCHEME_ORDER)
for sch in SCHEME_ORDER:
    for l, c in zip(LAYERS, dcols):
        rec(f"drift_large40_{sch}_{l}", H_large.loc[sch, c], "Stage2/logs/stage2_runs.csv[large,n=40]: mean(drift_<layer>) by layer_scheme")
        rec(f"drift_small40_{sch}_{l}", H_small.loc[sch, c], "Stage2/logs/stage2_runs.csv[small,n=40]: mean(drift_<layer>) by layer_scheme")
_lf = H_large.loc["full"]; _sf = H_small.loc["full"]
N["drift_large40_full_argmax_layer"] = {"value": LAYERS[int(np.argmax(_lf.values))], "source": "derived: argmax over drift_large40_full_*"}
N["drift_small40_full_argmax_layer"] = {"value": LAYERS[int(np.argmax(_sf.values))], "source": "derived: argmax over drift_small40_full_*"}
N["drift_large40_table_max_cell"] = {"value": f"bottleneck_only/encoder.4" if H_large.values.max() == H_large.loc["bottleneck_only", "drift_encoder.4"] else "?", "source": "derived: argmax over H_large"}
rec("drift_large40_table_max", H_large.values.max(), "derived: max over Stage2 large n=40 drift table")

# merged vs single-ID (each layer scheme)
for sch in SCHEME_ORDER:
    piv = lg[lg.layer_scheme == sch].pivot(index="start", columns="data_level", values="pooled_auc")
    for n in LEVELS:
        rec(f"stage2_large_{sch}_merged_n{n}", piv.loc["merged", n], f"Stage2/logs/stage2_configs.csv[large,{sch},start=merged]: pooled_auc")
        rec(f"stage2_large_{sch}_singles_mean_n{n}", piv.loc[IDS, n].mean(), f"Stage2/logs/stage2_configs.csv[large,{sch},start in ids]: mean(pooled_auc)")
        rec(f"stage2_large_{sch}_singles_min_n{n}", piv.loc[IDS, n].min(), f"Stage2/logs/stage2_configs.csv[large,{sch}]: min(pooled_auc) over single starts")
        rec(f"stage2_large_{sch}_singles_max_n{n}", piv.loc[IDS, n].max(), f"Stage2/logs/stage2_configs.csv[large,{sch}]: max(pooled_auc) over single starts")
        rec(f"stage2_large_{sch}_merged_minus_singles_n{n}", piv.loc["merged", n] - piv.loc[IDS, n].mean(), "derived: merged - mean(singles)")
        for m in IDS:
            rec(f"stage2_large_{sch}_{m}_n{n}", piv.loc[m, n], f"Stage2/logs/stage2_configs.csv[large,{sch},start={m}]: pooled_auc")
lf_full = lg[lg.layer_scheme == "full"].pivot(index="start", columns="data_level", values="pooled_auc")
lf_std = lg[lg.layer_scheme == "full"].pivot(index="start", columns="data_level", values="fold_auc_std")
lf_forget = lg[lg.layer_scheme == "full"].pivot(index="start", columns="data_level", values="source_auc_forget")
for m in STARTS:
    for n in LEVELS:
        rec(f"stage2_large_full_fold_std_{m}_n{n}", lf_std.loc[m, n], "Stage2/logs/stage2_configs.csv[large,full]: fold_auc_std")
    rec(f"stage2_large_full40_forget_{m}", lf_forget.loc[m, 40], "Stage2/logs/stage2_configs.csv[large,full,n=40]: source_auc_forget")

# Stage 3
for n in LEVELS:
    rec(f"stage3_pooled_auc_n{n}", c3.loc[n, "pooled_auc"], "Stage3/logs/stage3_configs.csv: pooled_auc")
    rec(f"stage3_fold_auc_mean_n{n}", c3.loc[n, "fold_auc_mean"], "Stage3/logs/stage3_configs.csv: fold_auc_mean")
    rec(f"stage3_fold_auc_std_n{n}", c3.loc[n, "fold_auc_std"], "Stage3/logs/stage3_configs.csv: fold_auc_std")
    rec(f"stage3_pooled_f1_n{n}", c3.loc[n, "pooled_best_f1"], "Stage3/logs/stage3_configs.csv: pooled_best_f1")
    rec(f"pretrain_value_n{n}", lg_curve.loc["full", n] - c3.loc[n, "pooled_auc"], "derived: stage2_large_auc_full_n - stage3_pooled_auc_n")
rec("stage3_runs", len(r3), "Stage3/logs/stage3_runs.csv: rows")
rec("stage3_failures", 0, "Stage3/logs/stage3_errors.jsonl: file absent")
rec("stage3_fold_auc_min_n5", r3[r3.data_level == 5]["target_auc"].min(), "Stage3/logs/stage3_runs.csv[n=5]: min(target_auc)")
rec("stage3_fold_auc_max_n40", r3[r3.data_level == 40]["target_auc"].max(), "Stage3/logs/stage3_runs.csv[n=40]: max(target_auc)")
_pv = {n: N[f"pretrain_value_n{n}"]["value"] for n in LEVELS}
rec("pretrain_value_argmax_level", max(_pv, key=_pv.get), "derived: argmax pretrain_value_n*")

rec("stage1_large_singles_mean_auc", s1l.loc[IDS, "pooled_auc"].mean(), "Stage1/logs/stage1_large_shift.csv: mean(pooled_auc) over single starts")
rec("stage1_large_merged_minus_singles", s1l.loc["merged", "pooled_auc"] - s1l.loc[IDS, "pooled_auc"].mean(), "derived: merged pooled_auc - mean(singles)")
rec("steps_ratio_40_over_5", N["stage2_steps_n40"]["value"] / N["stage2_steps_n5"]["value"], "derived: steps(n=40)/steps(n=5)")
for sch in ["encoder_only", "bottleneck_only"]:
    piv = lg[lg.layer_scheme == sch].pivot(index="start", columns="data_level", values="pooled_auc")
    rec(f"stage2_large_{sch}_singles_excl_id06_mean_n40", piv.loc[["id_00", "id_02", "id_04"], 40].mean(), f"Stage2/logs/stage2_configs.csv[large,{sch},n=40]: mean(pooled_auc) over id_00/id_02/id_04")

# Perfect folds
fold40 = r2[(r2["shift"] == "large") & (r2.data_level == 40) & (r2.layer_scheme == "full")]
rec("stage2_large_full40_folds_auc_eq_1", int((fold40["target_auc"] == 1.0).sum()), "Stage2/logs/stage2_runs.csv[large,n=40,full]: count(target_auc==1.0)")
rec("stage2_large_full40_folds_total", len(fold40), "Stage2/logs/stage2_runs.csv[large,n=40,full]: rows")
rec("stage2_large_full40_fold_auc_min", fold40["target_auc"].min(), "Stage2/logs/stage2_runs.csv[large,n=40,full]: min(target_auc)")
rec("stage2_large_full40_starts_with_zero_std", int((lf_std[40] == 0).sum()), "derived: count(fold_auc_std==0) at n=40 full")

# Scale
rec("total_training_runs", 5 + len(r2) + len(r3), "derived: 5 (stage0) + stage2 runs + stage3 runs")
rec("total_pretrained_models", 5, "Baseline/models: 5 files")
rec("stage1_small_pairs", len(s1s), "Stage1/logs/stage1_small_shift.csv: rows")
rec("stage1_large_models", len(s1l), "Stage1/logs/stage1_large_shift.csv: rows")
rec("stage1_large_folds_rows", len(s1f), "Stage1/logs/stage1_large_shift_folds.csv: rows")
rec("f1_trivial_floor", 2 * (2 / 3) / (1 + 2 / 3), "derived: F1 of predicting all abnormal at a 1:2 class ratio = 2*(2/3)/(1+2/3)")

# ============================================================================ fig3_rq1_shift_overview
# ============================================================================ Fig. 3: RQ1 overview (frozen evaluation under both domain shifts)
def _fig3_rq1_swarm(ax, x, ys, color, marker="o", zorder=5, min_gap_pt=7.2, step=0.045, max_k=6, preplaced=()):
    """Deterministic swarm: sort by value; when points are closer (in pt) than one marker diameter, offset alternately left/right so all 12 points are countable."""
    fig = ax.figure
    ax_w_in = ax.get_position().width * fig.get_figwidth()
    ax_h_in = ax.get_position().height * fig.get_figheight()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    px_per_x = 72 * ax_w_in / (x1 - x0); px_per_y = 72 * ax_h_in / (y1 - y0)
    placed = list(preplaced)   # already-placed points (e.g. the merged diamond) are also avoided
    for y in sorted(ys):
        for k in range(0, max_k + 1):
            for sgn in ((0,) if k == 0 else (1, -1)):
                xx = x + sgn * k * step
                if all(((xx - px) * px_per_x) ** 2 + ((y - py) * px_per_y) ** 2 >= min_gap_pt ** 2 for px, py in placed):
                    break
            else:
                continue
            break
        placed.append((xx, y))
    placed = placed[len(preplaced):]
    xs = np.array([q[0] for q in placed]); yy = np.array([q[1] for q in placed])
    ax.scatter(xs, yy, s=S_PT, color=color, marker=marker, edgecolor="white", linewidth=1.1, zorder=zorder)


def fig3_rq1(with_title):
    fig, axes = plt.subplots(2, 1, figsize=(COLW, 4.0), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.975, bottom=0.115, hspace=0.22)
    panels = [
        (axes[0], "auc_drop", "AUC drop", "{:.3f}", 0.0, (-0.15, 0.7)),
        (axes[1], "shift_ratio", "Threshold-shift ratio (×)", "{:.2f}×", 1.0, (0.0, 9.3)),
    ]
    axes[1].set_xlim(-0.6, 1.9)
    for ax, col, ylab, fmt, ref, ylim in panels:
        ax.set_ylim(*ylim)
        ax.axhline(ref, color=AXIS, linewidth=0.8, zorder=1)
        ms, ml = s1s[col].mean(), s1l[col].mean()
        # Mean line drawn below the points (zorder 2), never covering data; label to the right of the swarm
        for xx, m, lab in ((0, ms, "mean " + fmt.format(ms)), (1, ml, "mean " + fmt.format(ml))):
            ax.hlines(m, xx - 0.24, xx + 0.24, color=INK, linewidth=1.7, zorder=2)
            ax.annotate(lab, (xx + 0.28, m), va="center", ha="left", fontsize=8.5, color=INK, zorder=6)
        _fig3_rq1_swarm(ax, 0, s1s[col].values, BLUE)
        _fig3_rq1_swarm(ax, 1, s1l.loc[IDS, col].values, BLUE, preplaced=[(1.0, float(s1l.loc["merged", col]))])
        ax.scatter([1.0], [s1l.loc["merged", col]], s=S_PT, color=MERGED_COLOR, marker=MERGED_MARK,
                   edgecolor="white", linewidth=1.1, zorder=6)
        ax.set_ylabel(ylab)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Small shift\nMIMII ID → ID\n(12 pairs)", "Large shift\nMIMII → household\n(5 models)"])
    axes[1].annotate("1× = no shift", (1.88, 1.0), xytext=(0, 2), textcoords="offset points",
                     fontsize=8, color=MUTED, va="bottom", ha="right")
    handles = [
        Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=MS_PT,
               markeredgecolor="white", markeredgewidth=1.1, label="Single-ID model"),
        Line2D([], [], marker=MERGED_MARK, color=MERGED_COLOR, linestyle="", markersize=MS_PT,
               markeredgecolor="white", markeredgewidth=1.1, label="merged"),
        Line2D([], [], color=INK, linewidth=1.7, label="Mean"),
    ]
    legend_above(fig, handles, 3)
    suptitle(fig, with_title, "Figure · Frozen evaluation under the two domain shifts")
    return fig

# ============================================================================ fig4_source_vs_target
# ============================================================================ Fig. 4: source AUC and frozen target AUC (wide light bar + narrow dark bar)
def _fig4_bars_panel(ax, x, labels, src, tgt, light, highlight=None, rotation=0, wide=0.8, narrow=0.42):
    """At each x: wide light bar = source AUC, narrow dark bar = frozen target AUC, both from 0; when target > source the dark bar sticks out."""
    ax.bar(x, src, width=wide, color=light, zorder=3)
    ax.bar(x, tgt, width=narrow, color=BLUE, zorder=4)
    if highlight is not None:
        for xi, lab, s in zip(x, labels, src):
            if lab == highlight:
                ax.add_patch(Rectangle((xi - wide / 2, 0), wide, s, fill=False,
                                       edgecolor=MERGED_COLOR, linewidth=1.6, zorder=5))
    ax.axhline(0.5, color=INK, linewidth=0.9, linestyle="--", zorder=4)
    ax.set_xticks(x)
    if rotation:
        ax.set_xticklabels(labels, rotation=rotation, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(labels)
    if highlight is not None:
        for t in ax.get_xticklabels():
            if t.get_text() == highlight:
                t.set_color(MERGED_COLOR)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("AUC")


def fig4_bars(with_title):
    light = "#cde2fb"                        # wide light bar for source AUC (= SEQ_BLUE[1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COLW, 4.6))
    fig.subplots_adjust(left=0.17, right=0.94, top=0.94, bottom=0.09, hspace=0.66)   # <= single-column width including 0.15 in bleed

    # Top: small shift, 12 pairs sorted by (start, target), grouped in threes with gaps between groups
    d = s1s.sort_values(["source_model", "target_id"]).reset_index(drop=True)
    gap = 0.6
    x1 = np.array([i + (i // 3) * gap for i in range(len(d))], dtype=float)
    _fig4_bars_panel(ax1, x1, [t.replace("id_", "") for t in d["target_id"]], d["source_auc"].values, d["auc"].values, light, rotation=45)
    for gi, m in enumerate(IDS):
        xs_ = x1[gi * 3: gi * 3 + 3]
        ax1.annotate(f"start\n{m}", (xs_.mean(), -0.22), ha="center", va="top", fontsize=8, color=INK2, linespacing=1.15,
                     xycoords=("data", "axes fraction"), annotation_clip=False)
    ax1.set_xlim(x1[0] - 0.7, x1[-1] + 0.7)
    ax1.set_title("Small shift: 4 starts × 3 unseen IDs", loc="left")

    # Bottom: large shift, 5 starts -> household fan
    x2 = np.arange(len(STARTS), dtype=float)
    _fig4_bars_panel(ax2, x2, STARTS, b0.loc[STARTS, "auc"].values, s1l.loc[STARTS, "pooled_auc"].values, light,
                     highlight="merged")
    ax2.set_xlim(-0.7, len(STARTS) - 0.3)
    ax2.set_xlabel("Starting model")
    ax2.set_title("Large shift: 5 models → household fan", loc="left")

    handles = [Patch(color=light, label="Source AUC"),
               Patch(color=BLUE, label="Target AUC (frozen)"),
               Line2D([], [], color=INK, linestyle="--", linewidth=0.9, label="AUC = 0.5"),
               Patch(facecolor="none", edgecolor=MERGED_COLOR, linewidth=1.6, label="merged")]
    legend_above(fig, handles, ncol=2)   # 4 items in one row exceed single-column width (measured 20% over), so 2 rows
    suptitle(fig, with_title, "Figure · Source AUC and frozen target AUC")
    return fig

# ============================================================================ fig5_data_efficiency
# ============================================================================ Fig. 5: data efficiency (with no-pretraining baseline)
_FIG5_DATAEFF_MS = 5.0          # markersize of the four layer-scheme curves (same in legend)
_FIG5_DATAEFF_LW = 1.8
_FIG5_DATAEFF_MS_X = 5.5        # x marker of the no-pretraining baseline
_FIG5_DATAEFF_LW_X = 1.5


def _fig5_dataeff_panel(ax, curve, frozen, title, extra=None):
    xs = np.arange(len(LEVELS))
    for k, sch in enumerate(SCHEME_ORDER):   # earlier in order = higher zorder: in the small-shift panel Full (circle) and Decoder-only (square) nearly overlap; circle on top keeps both visible
        ys = [curve.loc[sch, n] for n in LEVELS]
        ax.plot(xs, ys, color=SCHEME_COLOR[sch], linewidth=_FIG5_DATAEFF_LW, marker=SCHEME_MARK[sch],
                markersize=_FIG5_DATAEFF_MS, markeredgecolor="white", markeredgewidth=1.0,
                zorder=3.5 - 0.1 * k)
    if extra is not None:
        ys = [extra.loc[n] for n in LEVELS]
        ax.plot(xs, ys, color=MUTED, linewidth=_FIG5_DATAEFF_LW_X, linestyle="--", marker="x",
                markersize=_FIG5_DATAEFF_MS_X, markeredgewidth=1.3, zorder=3)
    ax.axhline(frozen, color=MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels([f"{n} clips" for n in LEVELS]); ax.set_xlim(-0.3, 3.3)
    ax.set_ylabel("Target AUC"); ax.set_title(title, loc="left")


def _fig5_dataeff_seg_label(ax, fig, x0, y0, x1, y1, text, gap_pt=4.0, fs=8.0):
    """Place the increment text just above the segment midpoint: measure the text width, then lift it gap_pt above the highest point of the segment over that x span."""
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    t = ax.text(xm, ym, text, ha="center", va="bottom", fontsize=fs, color=BLUE, zorder=5)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    (xa, _), (xb, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    slope = (y1 - y0) / (x1 - x0)
    line_max = max(y0 + slope * (xa - x0), y0 + slope * (xb - x0))
    (_, ya), (_, yb) = inv.transform([(0, 0), (0, gap_pt / 72 * fig.dpi)])
    t.set_position((xm, line_max + (yb - ya)))
    return t


def fig5_dataeff(with_title):
    fig, axes = plt.subplots(2, 1, figsize=(COLW, 4.8))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.94, bottom=0.085, hspace=0.3)
    _fig5_dataeff_panel(axes[0], sm_curve, N["stage1_small_mean_auc"]["value"], "Small shift (mean of 12 pairs)")
    _fig5_dataeff_panel(axes[1], lg_curve, N["stage1_large_mean_auc"]["value"], "Large shift (mean of 5 starts, pooled)",
                        extra=c3["pooled_auc"])
    axes[0].set_ylim(0.3, 1.03)
    axes[1].set_ylim(0.3, 1.08)
    axes[1].set_xlabel("Target-domain normal clips")

    # Three increments of the Full curve + hollow ring at 20 clips (reading aid, covers no line/point)
    ax = axes[1]
    f_ = lg_curve.loc["full"]
    for (a, b), key in zip(((5, 10), (10, 20), (20, 40)), ("5_10", "10_20", "20_40")):
        d = N[f"stage2_large_full_delta_{key}"]["value"]
        i = LEVELS.index(a)
        _fig5_dataeff_seg_label(ax, fig, i, f_[a], i + 1, f_[b], f"+{d:.3f}")
    ax.plot([LEVELS.index(20)], [f_[20]], marker="o", markersize=12, markerfacecolor="none",
            markeredgecolor=BLUE, markeredgewidth=1.2, linestyle="", zorder=4)

    # Legend: two rows, all outside above the plot area. Lower row = grey baselines; upper row = four layer schemes
    h_grey = [Line2D([], [], color=MUTED, linewidth=_FIG5_DATAEFF_LW_X, linestyle="--", marker="x",
                     markersize=_FIG5_DATAEFF_MS_X, markeredgewidth=1.3, label="No pretraining (random init.)"),
              Line2D([], [], color=MUTED, linewidth=1.0, linestyle=":", label="Frozen baseline (Stage 1)")]
    h_sch = [Line2D([], [], color=SCHEME_COLOR[s], linewidth=_FIG5_DATAEFF_LW, marker=SCHEME_MARK[s],
                    markersize=_FIG5_DATAEFF_MS, markeredgecolor="white", markeredgewidth=1.0, label=SCHEME_CN[s])
             for s in SCHEME_ORDER]
    leg1 = legend_above(fig, h_grey, ncol=1, y=1.0)
    fig.canvas.draw()
    bb = leg1.get_window_extent().transformed(fig.transFigure.inverted())
    legend_above(fig, h_sch, ncol=2, y=bb.y1 + 0.004, handlelength=1.3, columnspacing=0.9)   # English labels are longer: two rows
    suptitle(fig, with_title, "Figure · Data efficiency")
    return fig

# ============================================================================ fig6_gain_vs_forget
# ============================================================================ Fig. 6: gain vs forgetting (two rows, shared x)
_FIG6_GAINFORGET_SIZE = {5: 18, 10: 34, 20: 54, 40: 80}   # the only figure that encodes data level by marker area
_FIG6_GAINFORGET_REGION = dict(facecolor="#eef3fa", edgecolor="#9ec5f4", linewidth=0.8)


def _fig6_gainforget_legend_rows(fig, rows, y0=1.0, gap=0.0):
    """Stack several legend rows bottom-up above the figure; rows are given bottom to top as (handles, ncol, kw).
    Each row is its own fig.legend, drawn once to measure height before placing the next; handlelength slightly narrowed so total width stays within COLW."""
    y = y0
    for handles, ncol, kw in rows:
        lg = legend_above(fig, handles, ncol, y=y, handlelength=1.0, columnspacing=0.7, **kw)
        fig.canvas.draw()
        bb = lg.get_window_extent().transformed(fig.transFigure.inverted())
        y = bb.y1 + gap


def _fig6_gainforget_panel(ax, df, ycol, title):
    ax.grid(True, axis="both", color=GRID, linewidth=0.6)
    # Low-cost, high-gain region: forgetting < FORGET_T and gain > GAIN_T; the rectangle is clipped by the axes, leaving only its right and bottom edges
    ax.add_patch(Rectangle((-1.0, GAIN_T), 1.0 + FORGET_T, 5.0, zorder=1, **_FIG6_GAINFORGET_REGION))
    ax.axhline(0, color=AXIS, linewidth=0.9, zorder=2)
    ax.axvline(0, color=AXIS, linewidth=0.9, zorder=2)
    # Large markers first, small ones last, so small markers are not hidden
    for n in reversed(LEVELS):
        for sch in SCHEME_ORDER:
            sub = df[(df.layer_scheme == sch) & (df.data_level == n)]
            ax.scatter(sub["source_auc_forget"], sub[ycol], s=_FIG6_GAINFORGET_SIZE[n], color=SCHEME_COLOR[sch],
                       marker=SCHEME_MARK[sch], edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)
    ax.set_xlim(-0.05, 0.80); ax.set_ylim(-0.25, 0.85)
    ax.set_xticks(np.arange(0, 0.81, 0.2)); ax.set_yticks(np.arange(-0.2, 0.81, 0.2))
    ax.set_ylabel("Target-domain gain"); ax.set_title(title, loc="left")


def fig6_gainforget(with_title):
    fig, axes = plt.subplots(2, 1, figsize=(COLW, 5.0), sharex=True)
    fig.subplots_adjust(left=0.205, right=0.937, top=0.94, bottom=0.09, hspace=0.20)
    _fig6_gainforget_panel(axes[0], sm, "target_auc_gain", f"Small shift ({len(sm)} configurations)")
    _fig6_gainforget_panel(axes[1], lg, "pooled_auc_gain", f"Large shift ({len(lg)} configurations)")
    axes[1].set_xlabel("Source-domain forgetting")
    h_scheme = [Line2D([], [], linestyle="", marker=SCHEME_MARK[s], color=SCHEME_COLOR[s], markersize=6,
                       markeredgecolor="white", markeredgewidth=0.8, label=SCHEME_CN[s]) for s in SCHEME_ORDER]
    h_level = [Line2D([], [], linestyle="", marker="o", color=MUTED, markersize=np.sqrt(_FIG6_GAINFORGET_SIZE[n]),
                      markeredgecolor="white", markeredgewidth=0.8, label=f"{n} clips") for n in LEVELS]
    h_region = [Patch(label=f"Low-cost, high-gain region\n(forgetting < {FORGET_T:.2f}, gain > {GAIN_T:.2f})", **_FIG6_GAINFORGET_REGION)]
    _fig6_gainforget_legend_rows(fig, [(h_region, 1, {}), (h_level, 4, {"handleheight": 1.2}), (h_scheme, 2, {})])
    suptitle(fig, with_title, "Figure · Gain versus forgetting")
    return fig

# ============================================================================ fig7_merged_vs_singles
# ============================================================================ Fig. 7: merged vs single-ID starts (large shift, 2x2 by layer scheme)
def _fig7_merged_panel(ax, sch, xs, band_color):
    """One layer scheme: grey band (min-max of the 4 single-ID starts) + merged red line + id_06 grey dashed line."""
    piv = lg[lg.layer_scheme == sch].pivot(index="start", columns="data_level", values="pooled_auc")
    lo = [piv.loc[IDS, n].min() for n in LEVELS]
    hi = [piv.loc[IDS, n].max() for n in LEVELS]
    ax.fill_between(xs, lo, hi, color=band_color, alpha=0.85, linewidth=0, zorder=1)
    ax.plot(xs, [piv.loc["id_06", n] for n in LEVELS], color=INK2, linewidth=1.1, linestyle="--",
            marker="v", markersize=3.5, zorder=3)
    ax.plot(xs, [piv.loc["merged", n] for n in LEVELS], color=MERGED_COLOR, linewidth=2, marker=MERGED_MARK,
            markersize=5, markeredgecolor="white", markeredgewidth=0.9, zorder=4)
    ax.set_title(SCHEME_CN[sch], loc="left")


def fig7_merged(with_title):
    band_color = "#d9d8d2"
    fig, axes = plt.subplots(2, 2, figsize=(COLW, 3.7), sharey="row", sharex=True, layout="constrained")
    xs = np.arange(len(LEVELS))
    for ax, sch in zip(axes.ravel(), SCHEME_ORDER):
        _fig7_merged_panel(ax, sch, xs, band_color)
        ax.set_xticks(xs)
        ax.set_xlim(-0.35, 3.35)
    for ax in axes[1, :]:
        ax.set_xticklabels([str(n) for n in LEVELS])   # unit "clips" goes in the x-axis title to avoid tick overlap in the narrow figure
    axes[0, 0].set_ylim(0.65, 1.01)
    axes[0, 0].set_yticks([0.7, 0.8, 0.9, 1.0])
    axes[1, 0].set_ylim(0.42, 0.80)
    axes[1, 0].set_yticks([0.5, 0.6, 0.7, 0.8])
    lab_color = plt.rcParams["axes.labelcolor"]
    fig.supylabel("Pooled target AUC", fontsize=8.5, color=lab_color)
    fig.supxlabel("Fine-tuning data (target normal clips)", fontsize=8.5, color=lab_color)
    # Legend: one row of three items (~3.0 in, fits single-column width); merged's full name ("mixed pretraining on four IDs") is in the LaTeX caption.
    handles = [
        Patch(facecolor=band_color, alpha=0.85, edgecolor="none", label="Single-ID range"),
        Line2D([], [], color=MERGED_COLOR, linewidth=2, marker=MERGED_MARK, markersize=5,
               markeredgecolor="white", markeredgewidth=0.9, label="merged"),
        Line2D([], [], color=INK2, linewidth=1.1, linestyle="--", marker="v", markersize=3.5, label="id_06"),
    ]
    leg = legend_above(fig, handles, ncol=3, y=1.0)
    # Title (original version only): suptitle aligns at the text top, and the default +0.025 gap is smaller than the 10 pt text height,
    # so it would overlap the legend; here the title is placed 0.06 (~0.22 in) above the actual legend top.
    fig.canvas.draw()
    leg_top = leg.get_window_extent().transformed(fig.transFigure.inverted()).y1
    suptitle(fig, with_title, "Figure · merged and single-ID starting models", y=leg_top + 0.06)
    return fig

# ============================================================================ figA1_weight_drift
# ============================================================================ Fig. A1: relative weight drift heatmap (appendix)
def _figA1_drift_runs_per_cell(shift):
    """How many runs each cell of H_small / H_large averages: same filter as H_* in the prelude
    (shift, data_level==40, grouped by layer_scheme). Returns that count when it is identical across the four schemes."""
    counts = r2[(r2["shift"] == shift) & (r2.data_level == 40)].groupby("layer_scheme").size().reindex(SCHEME_ORDER)
    assert counts.notna().all() and counts.nunique() == 1, counts
    return int(counts.iloc[0])


def _figA1_drift_panel(fig, rect, Hm, title, vmax):
    """One heatmap: 4 layer schemes x 6 layers; no grid, no frame, tick length 0. Returns (ax, im)."""
    ax = fig.add_axes(rect)
    ax.grid(False)
    im = ax.imshow(Hm.values, cmap=CMAP_BLUE, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(6))
    ax.set_xticklabels([l.replace("encoder.", "enc.").replace("decoder.", "dec.") for l in LAYERS], fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels([SCHEME_CN[s].replace("-only", "-\nonly") for s in SCHEME_ORDER], fontsize=8.5, linespacing=0.95)   # English labels wrapped onto two lines to leave room for cells
    for i in range(4):
        for j in range(6):
            v = Hm.values[i, j]
            if v == 0:
                ax.text(j, i, "frozen", ha="center", va="center", fontsize=7.5, color=MUTED)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * vmax else INK)
    ax.axvline(2.5, color="white", linewidth=2.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=3)
    return ax, im


def figA1_drift(with_title):
    vmax = max(H_small.values.max(), H_large.values.max())
    n_small, n_large = _figA1_drift_runs_per_cell("small"), _figA1_drift_runs_per_cell("large")
    W, H = COLW, 3.66
    fig = plt.figure(figsize=(W, H))
    fx, fy = (lambda x: x / W), (lambda y: y / H)          # inches -> figure coordinates
    L, R, PH = 0.66, W - 0.02, 1.05                          # heatmap left/right edges and height (inches)
    tops = [3.40, 1.95]                                      # top edges of the two heatmaps (inches)
    panels = [(H_small, f"Small shift (n=40, mean of {n_small} runs per cell)"),
              (H_large, f"Large shift (n=40, mean of {n_large} runs per cell)")]
    im = None
    for top, (Hm, title) in zip(tops, panels):
        _, im = _figA1_drift_panel(fig, [fx(L), fy(top - PH), fx(R - L), fy(PH)], Hm, title, vmax)
        fig.text(0.0, fy(top + 0.05), title, ha="left", va="bottom", fontsize=9, color=INK)   # title starts at the figure's left edge so the English title does not overrun the heatmap's right edge
    # Encoder / Decoder group labels (secondary info, MUTED): only below the bottom panel's tick labels
    fig.text(fx(L + 0.25 * (R - L)), fy(0.70), "Encoder", ha="center", va="top", fontsize=8, color=MUTED)
    fig.text(fx(L + 0.75 * (R - L)), fy(0.70), "Decoder", ha="center", va="top", fontsize=8, color=MUTED)
    # Shared colorbar: horizontal, below the bottom panel (keeps the heatmaps as wide as possible)
    cax = fig.add_axes([fx(L + 0.2 * (R - L)), fy(0.41), fx(0.6 * (R - L)), fy(0.09)])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    # Upright string; in Hiragino the double bar and Delta are full-width (1 em) and too wide, so this label uses DejaVu Sans
    cb.set_label(r"$\|\Delta W\|\,/\,\|W\|$", fontsize=8.5, labelpad=3)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=8, pad=2)
    suptitle(fig, with_title, "Figure · Relative weight drift")
    return fig

# ============================================================================ figA2_per_fold_auc
# ============================================================================ Fig. A2: per-fold AUC (appendix)
def _figA2_perfold_handles():
    mk = dict(linestyle="", markersize=MS_PT, markeredgecolor="white", markeredgewidth=1.1)
    return [Line2D([], [], marker="o", color=BLUE, label="Single-ID start", **mk),
            Line2D([], [], marker=MERGED_MARK, color=MERGED_COLOR, label="merged", **mk),
            Line2D([], [], marker="o", color=ORANGE, label="No pretraining", **mk),
            Line2D([], [], color=INK, linewidth=1.7, label="Mean")]


def figA2_perfold(with_title):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(COLW, 2.85))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.885, bottom=0.11, hspace=0.80)

    # ---- Top: pretrained + full fine-tuning (n=40), per-fold AUC for the 5 starts
    ax = ax_top
    for i, m in enumerate(STARTS):
        ys = fold40[fold40.start == m]["target_auc"].values
        if m == "merged":
            strip(ax, i, ys, MERGED_COLOR, marker=MERGED_MARK, jitter=0.12, size=S_PT, seed=i + 10)
        else:
            strip(ax, i, ys, BLUE, jitter=0.12, size=S_PT, seed=i + 10)
    ax.set_xticks(range(len(STARTS))); ax.set_xticklabels(STARTS); ax.set_xlim(-0.5, len(STARTS) - 0.5)
    for t in ax.get_xticklabels():
        if t.get_text() == "merged":
            t.set_color(MERGED_COLOR)
    ax.set_xlabel("Starting model")
    n_perfect = N["stage2_large_full40_folds_auc_eq_1"]["value"]
    n_total = N["stage2_large_full40_folds_total"]["value"]
    ax.set_title(f"Pretrained + full fine-tuning (n=40):\n{n_perfect}/{n_total} folds with AUC = 1.000", loc="left")

    # ---- Bottom: no pretraining (random init.), per-fold AUC at each data level
    ax = ax_bot
    for i, n in enumerate(LEVELS):
        ys = r3[r3.data_level == n]["target_auc"].values
        strip(ax, i, ys, ORANGE, jitter=0.09, size=S_PT, seed=i + 20)
        # Round 6: mean label at the right end of the mean line with a white box, so it does not cover points in this or adjacent groups
        m_ = ys.mean()
        ax.hlines(m_, i - 0.2, i + 0.2, color=INK, linewidth=1.7, zorder=4)
        ax.annotate(f"{m_:.3f}", (i + 0.24, m_), va="center", ha="left", fontsize=8, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9))
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels([f"{n} clips" for n in LEVELS])
    ax.set_xlim(-0.5, 3.85)
    ax.set_xlabel("Target-domain normal clips")
    ax.set_title("No pretraining (random init.)", loc="left")
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate("chance 0.5", (3.85, 0.5), xytext=(-2, 2), textcoords="offset points",
                va="bottom", ha="right", fontsize=8, color=MUTED)

    for ax in (ax_top, ax_bot):
        ax.set_ylim(0.42, 1.03)
    fig.supylabel("Per-fold AUC", fontsize=8.5, color="#3d3c39", x=0.02)   # per-fold composition (6 normal + 12 abnormal) is in the caption

    legend_above(fig, _figA2_perfold_handles(), ncol=4, handlelength=0.8, handletextpad=0.3, columnspacing=0.6)
    suptitle(fig, with_title, "Figure · Per-fold AUC")
    return fig


print("Generating figures:")
save_both(fig3_rq1, "fig3_rq1_shift_overview")
save_both(fig4_bars, "fig4_source_vs_target")
save_both(fig5_dataeff, "fig5_data_efficiency")
save_both(fig6_gainforget, "fig6_gain_vs_forget")
save_both(fig7_merged, "fig7_merged_vs_singles")
save_both(figA1_drift, "figA1_weight_drift")
save_both(figA2_perfold, "figA2_per_fold_auc")


# ============================================================================ LaTeX tables (data-generated)
def tex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def write_tab(name, body):
    (REP_TAB / f"{name}.tex").write_text(body, encoding="utf-8")


def fmt4(v):
    return f"{v:.4f}"


def fmt3s(v):
    return f"{v:+.3f}"


# Architecture and parameter counts
layer_dims = [(320, 64), (64, 64), (64, 8), (8, 64), (64, 64), (64, 320)]
params = {l: i * o + o for l, (i, o) in zip(LAYERS, layer_dims)}
for l in LAYERS:
    rec(f"params_layer_{l}", params[l], "derived: in*out+out from code/configs/config.yaml encoder_dims/decoder_dims")
rec("params_total", sum(params.values()), "derived: sum of layer params")


# Experiment scale
write_tab("tab_scale", "\n".join([
    "\\begin{tabular}{@{}l l r@{}}", "\\toprule",
    "Stage & Content & Runs \\\\", "\\midrule",
    "0 Pretraining & 4 single-ID + 1 mixed & 5 \\\\",
    f"1 Frozen evaluation & {N['stage1_small_pairs']['value']} small + {N['stage1_large_models']['value']} large & 0 \\\\",
    f"2 Fine-tuning & {N['stage2_configs']['value']} configs: 192 + 80 $\\times$ 10 folds & {N['stage2_runs']['value']} \\\\",
    f"3 No pretraining & 4 levels $\\times$ 10 folds & {N['stage3_runs']['value']} \\\\",
    "\\midrule", f"Total & 0 failures & {N['total_training_runs']['value']} \\\\",
    "\\bottomrule", "\\end{tabular}"]))




def pivot_tab(name, piv, fmt, caption_cols=LEVELS):
    rows = [f"{SCHEME_CN[s]} & " + " & ".join(fmt(piv.loc[s, n]) for n in caption_cols) + " \\\\" for s in SCHEME_ORDER]
    write_tab(name, "\n".join([
        "\\begin{tabular}{l " + "r " * len(caption_cols) + "}", "\\toprule",
        "Layer scheme & " + " & ".join(f"{n} clips" for n in caption_cols) + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))


pivot_tab("tab_a4_large_fold_std", lg_fstd, fmt4)

# Main-text table: n=40 gain and forgetting + parameters
rows = []
for sch in SCHEME_ORDER:
    p = N[f"params_trainable_{sch}"]["value"]
    rows.append(f"{SCHEME_CN[sch]} & {p:,} ({100 * p / sum(params.values()):.0f}\\%) & {sm_gain.loc[sch, 40]:+.3f} & {lg_gain.loc[sch, 40]:+.3f} & {sm_forget.loc[sch, 40]:.3f} & {lg_forget.loc[sch, 40]:.3f} \\\\")
write_tab("tab_gain_forget40", "\n".join([
    "\\begin{tabular}{@{}l r r r r r@{}}", "\\toprule",
    " & & \\multicolumn{2}{c}{Gain} & \\multicolumn{2}{c}{Forgetting} \\\\",
    "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
    "Scheme & Params & Small & Large & Small & Large \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))

# Fold std (appendix table): pivot_tab is still used for tab_a4_large_fold_std
def drift_tab(name, H):  # kept for verification; the report no longer outputs the drift table since round 3
    rows = [f"{SCHEME_CN[s]} & " + " & ".join(("frozen" if H.loc[s, c] == 0 else f"{H.loc[s, c]:.3f}") for c in dcols) + " \\\\" for s in SCHEME_ORDER]
    write_tab(name, "\n".join([
        "\\begin{tabular}{l " + "r " * 6 + "}", "\\toprule",
        "Scheme & " + " & ".join("\\texttt{" + l.replace("encoder.", "enc.").replace("decoder.", "dec.") + "}" for l in LAYERS) + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))





# A8: fold std (full, start x data level)
rows = [f"{tex_escape(m)} & " + " & ".join(fmt4(lf_std.loc[m, n]) for n in LEVELS) + " \\\\" for m in STARTS]
write_tab("tab_a8_fold_std", "\n".join([
    "\\begin{tabular}{l r r r r}", "\\toprule",
    "Starting model & " + " & ".join(f"{n} clips" for n in LEVELS) + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))


with open(ROOT / "report" / "numbers_audit.json", "w", encoding="utf-8") as fh:
    json.dump(N, fh, ensure_ascii=False, indent=1)

print(f"\n✅ 7 figures x 2 versions (plus 1 each from make_fan_photo.py / make_model_diagram.py) -> {ORIG.relative_to(ROOT)}, {CLEAN.relative_to(ROOT)}; clean PDFs copied to {REP_FIG.relative_to(ROOT)}")
print(f"   {len(list(REP_TAB.glob('*.tex')))} tables -> {REP_TAB.relative_to(ROOT)}; {len(N)} registered numbers -> report/numbers_audit.json; font {FONT}")
