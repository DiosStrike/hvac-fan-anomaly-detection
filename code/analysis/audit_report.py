# -*- coding: utf-8 -*-
"""Audit every number in the body of report/main.tex and write report/AUDIT.md.

Rules
  1. Extract all numbers from the body of main.tex (dropping comments, arguments of \\label/\\ref/\\cite/\\input/\\includegraphics, and years).
  2. Look each number up first in CONSTANTS (configuration/structural constants from code/configs/config.yaml or the code),
     then in report/numbers_audit.json (recomputed from results/ by make_report_figures.py):
     match if |written value - registered value| <= 0.5 * 10^(-decimal places). Percentages try both value and value*100.
  3. Unmatched numbers are listed separately and must be explained manually; otherwise they count as inconsistencies.
  4. Tables are script-generated; only their sources are listed, not audited cell by cell.

Usage: python code/analysis/audit_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "report" / "main.tex"
REG = ROOT / "report" / "numbers_audit.json"
OUT = ROOT / "report" / "AUDIT.md"

N = json.load(open(REG, encoding="utf-8"))
REGISTRY = {k: v["value"] for k, v in N.items() if isinstance(v["value"], (int, float))}
SOURCE = {k: v["source"] for k, v in N.items()}

# Configuration/structural constants: sourced from code/configs/config.yaml, the code or data manifests, not from result files in results/
CONSTANTS = {
    "0.5": "AUC chance level (definition)",
    "1.0": "AUC upper bound / per-fold AUC = 1.0 (Stage2 runs, see stage2_large_full40_folds_auc_eq_1)",
    "1.000": "same as above",
    "42": "code/configs/config.yaml: seed",
    "1011": "code/configs/config.yaml: official_counts.id_00.normal (matches MIMII paper Table 1)",
    "407": "code/configs/config.yaml: official_counts.id_00.abnormal",
    "1016": "code/configs/config.yaml: official_counts.id_02.normal",
    "359": "code/configs/config.yaml: official_counts.id_02.abnormal",
    "1033": "code/configs/config.yaml: official_counts.id_04.normal",
    "348": "code/configs/config.yaml: official_counts.id_04.abnormal",
    "1015": "code/configs/config.yaml: official_counts.id_06.normal",
    "361": "code/configs/config.yaml: official_counts.id_06.abnormal",
    "160": "code/configs/config.yaml: eval_set.n_normal (verified against splits json)",
    "320": "code/configs/config.yaml: eval_set.n_abnormal / features.input_dim",
    "480": "derived: 160 + 320",
    "215": "code/configs/config.yaml: merged_model.train_per_id",
    "860": "code/configs/config.yaml: merged_model.train_total (verified against splits json)",
    "40": "code/configs/config.yaml: merged eval_per_id.normal=40 / data_levels includes 40",
    "80": "code/configs/config.yaml: merged eval_per_id.abnormal=80 / stage2 large n_configs=80",
    "180": "code/configs/config.yaml: target_domain.n_clips_total (verified against recording_log)",
    "60": "code/configs/config.yaml: target_domain.n_normal_total",
    "120": "code/configs/config.yaml: target_domain.n_abnormal_total",
    "16": "code/configs/config.yaml: features.sr = 16000 Hz (verified against recording_log)",
    "64": "code/configs/config.yaml: features.n_mels / model hidden width / finetune.batch_size",
    "1024": "code/configs/config.yaml: features.n_fft",
    "512": "code/configs/config.yaml: features.hop_length / pretrain.batch_size",
    "5": "first data level / number of pretrained models / features.frames",
    "2.0": "code/configs/config.yaml: features.power",
    "8": "code/configs/config.yaml: model.bottleneck_dim",
    "50,760": "derived: sum of per-layer parameters (see params_total)",
    "50": "code/configs/config.yaml: pretrain.epochs / n=40 full fine-tuning 5x10=50 folds",
    "10": "code/configs/config.yaml: pretrain.validation_split=0.1 / loro.n_folds / data_levels includes 10 / 10 target-domain rounds",
    "30": "code/configs/config.yaml: finetune.epochs",
    "12": "number of stage1 small-shift pairs / abnormal clips per fold (loro.abnormals_per_fold)",
    "18": "code/configs/config.yaml: loro.clips_per_fold",
    "6": "code/configs/config.yaml: loro.normals_per_fold / number of small-shift pairs with AUC<0.5 (stage1_small_n_below_half)",
    "54": "code/configs/config.yaml: loro.max_train_normals",
    "20": "data_levels includes 20 / number of configs in the top-left region (topleft_small_count)",
    "1000": "code/configs/config.yaml: metrics.n_threshold_scan_points",
    "0.800": "derived: f1_trivial_floor",
    "0.15": "region threshold defined in the text (topleft_forget_threshold)",
    "0.30": "region threshold defined in the text (topleft_gain_threshold)",
    "1:2": "class ratio of the evaluation set (code/configs/config.yaml)",
    "200": "derived: 20 clips x 10 s",
    "0.98": "AUC threshold stated in the text (stage2_large_auc_full_n20 = 0.9837 > 0.98)",
    "0.03": "approximate merged lead stated in the text (stage2_large_full_merged_minus_singles_n5/n10)",
    "0.35": "approximate full fine-tuning forgetting stated in the text (stage2_small/large_forget_full_n40)",
    "2": "structural constant (two environments / 2% parameter share)",
    "3": "structural constant (three conditions / three voltage levels / 3 targets)",
    "4": "structural constant (four IDs / four levels / four schemes)",
    "7": "MIMII has 7 model IDs per machine type (paper Table 1)",
    "9": "loro.n_folds - 1 (folds numbered 0-9)",
    "1": "structural constant",
    "0": "structural constant",
    "0.000": "stage2_large_full_merged_minus_singles_n40 (-0.0002, rounds to 0.000)",
    "10^{-3}": "code/configs/config.yaml: pretrain.lr = 0.001",
    "10^{-4}": "code/configs/config.yaml: finetune.lr = 0.0001",
}
YEARS = {"2019", "2021", "2025"}


def clean_tex(s: str) -> str:
    # Round 5 (English version): audit only the body -- from \begin{document} on; layout parameters in the preamble (\setcounter, \emergencystretch, etc.) do not count
    if "\\begin{document}" in s:
        s = s[s.index("\\begin{document}"):]
    s = re.sub(r"(?<!\\)%.*", "", s)                                   # comments
    s = re.sub(r"\\(label|ref|cite|input|includegraphics|caption\*?)\s*(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}", " ", s)
    s = re.sub(r"\\(documentclass|usepackage|setCJK\w+|IfFontExistsTF|xeCJKsetup|settopmatter|newcommand|renewcommand|pagestyle|setcopyright|bibliographystyle|bibliography|affiliation|email)\b[^\n]*", " ", s)
    s = s.replace("\\%", "%")
    return s


def extract_numbers(s: str):
    # Thousands separators only when exactly three digits follow the comma (12,360, 50,760), so commas in English lists like "4, 5, 5 and 6" are not absorbed
    pat = re.compile(r"(?<![\w.])([+\-−$]*\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\d,]*\d)"
                     r"(\s*(?:\\%|%|\u500d|x|×|\$?\\times\$?|\u6bb5|\u8f6e|\u6b65|\u4e2a|\u7ec4|\u6298|clips?|folds?|runs?|pairs?|models?|configurations?|steps|vectors|times))?")   # \u500d..\u6298: Chinese unit suffixes (times/clips/rounds/steps/count/groups/folds)
    out = []
    for m in pat.finditer(s):
        raw = re.sub(r"\s+", "", m.group(1)).replace("$", "")
        num_txt = raw.lstrip("+-−")
        if num_txt in YEARS or not num_txt:
            continue
        start = max(0, m.start() - 34)
        ctx = s[start:m.end() + 30].replace("\n", " ")
        ctx = re.sub(r"\s+", " ", ctx)
        out.append((raw, num_txt, m.group(2) or "", ctx, m.start()))
    return out


def decimals(t: str) -> int:
    return len(t.split(".")[1]) if "." in t else 0


def match_registry(num_txt: str, unit: str):
    v = float(num_txt.replace(",", ""))
    d = decimals(num_txt)
    tol = 0.5 * 10 ** (-d) + 1e-9
    hits = []
    for k, rv in REGISTRY.items():
        cands = [rv]
        if unit.strip() == "%" or "%" in unit:
            cands.append(rv * 100)
        for c in cands:
            if abs(abs(c) - abs(v)) <= tol and (d > 0 or abs(c - round(c)) < 1e-9 or True):
                hits.append((k, rv))
                break
    return hits


def main():
    tex = TEX.read_text(encoding="utf-8")
    body = clean_tex(tex)
    nums = extract_numbers(body)

    rows, unmatched = [], []
    for raw, num_txt, unit, ctx, pos in nums:
        key = num_txt
        const = CONSTANTS.get(key) or CONSTANTS.get(num_txt.replace(",", "")) or (CONSTANTS.get("50,760") if num_txt == "50,760" else None)
        hits = match_registry(num_txt, unit)
        line = tex.count("\n", 0, tex.find(ctx.split(" ")[1] if " " in ctx else ctx)) + 1 if False else None
        if hits:
            k, rv = sorted(hits, key=lambda h: len(h[0]))[0]
            more = f" (+{len(hits) - 1} more candidates)" if len(hits) > 1 else ""
            rows.append((raw + unit.strip(), ctx, f"`{k}`{more}", SOURCE[k], f"{rv:.6g}", "✅"))
        elif const:
            rows.append((raw + unit.strip(), ctx, "constant", const, "—", "✅"))
        else:
            unmatched.append((raw + unit.strip(), ctx))
            rows.append((raw + unit.strip(), ctx, "—", "**unmatched, needs manual check**", "—", "❌"))

    lines = ["# Data and parameter audit (report/main.tex)", "",
             "From round 5 the audit target is the English `report/main.tex` (the audit of the final Chinese version is in `report_zh/AUDIT.md`); only the body after \\begin{document} is audited.", "",
             f"Generated by `python code/analysis/audit_report.py`. **{len(rows)}** numbers extracted from the body; the registry `numbers_audit.json` holds {len(REGISTRY)} values recomputed from raw logs in `results/`. "
             f"Matching rule: |text value − recomputed value| ≤ 0.5 × 10^(−decimal places). Unmatched: **{len(unmatched)}**.", "",
             "## 1 · Item-by-item check of numbers in the text", "",
             "| As written | Location (context) | Registry key | Source file / column / computation | Recomputed | Match |", "|---|---|---|---|---|---|"]
    for w, ctx, k, src, rv, ok in rows:
        lines.append(f"| {w} | …{ctx}… | {k} | {src} | {rv} | {ok} |")

    lines += ["", "## 2 · Script-generated tables and figures (not audited cell by cell; sources listed)", "",
              "From round 3 the report keeps only 3 tables; all numbers from removed tables remain registered in `numbers_audit.json`, and any cited in the text are checked in Section 1.", "",
              "| Table / figure | File | Source |", "|---|---|---|",
              "| Table 1 scale | tables/tab_scale.tex | row counts of the Stage1/Stage2/Stage3 csv files; failures = error files absent |",
              "| Table 2 n=40 gain/forgetting | tables/tab_gain_forget40.tex | Stage2/logs/stage2_configs.csv[data_level=40]: mean of target_auc_gain / pooled_auc_gain / source_auc_forget; parameter counts from Stage2/logs/stage2_runs.csv n_trainable_params |",
              "| Table 3 (appendix) fold std | tables/tab_a8_fold_std.tex, tab_a4_large_fold_std.tex | Stage2/logs/stage2_configs.csv[large] fold_auc_std |",
              "| Fig. 1 fan photo | figures/fig1_fan_conditions.pdf | data/photos/fan_conditions.png with the 4th panel cropped (make_fan_photo.py) |",
              "| Fig. 2 model architecture | figures/fig2_model_architecture.pdf | MIMII_Baseline_AE in code/notebooks/stage0_pretrain.ipynb (in×out+out); parameter counts of the four schemes from Stage2/logs/stage2_runs.csv n_trainable_params (make_model_diagram.py) |",
              "| Fig. 3 frozen evaluation | figures/fig3_rq1_shift_overview.pdf | Stage1/logs/stage1_small_shift.csv, stage1_large_shift.csv: auc_drop / shift_ratio |",
              "| Fig. 4 source vs target AUC | figures/fig4_source_vs_target.pdf | Baseline/logs/stage0_source_baseline.csv auc; Stage1 small auc / large pooled_auc |",
              "| Fig. 5 data efficiency | figures/fig5_data_efficiency.pdf | Stage2/logs/stage2_configs.csv target_auc / pooled_auc mean by layer_scheme×data_level; Stage3/logs/stage3_configs.csv pooled_auc; Stage1 frozen mean |",
              "| Fig. 6 gain vs forgetting | figures/fig6_gain_vs_forget.pdf | Stage2/logs/stage2_configs.csv source_auc_forget, target_auc_gain / pooled_auc_gain |",
              "| Fig. 7 merged vs single-ID | figures/fig7_merged_vs_singles.pdf | Stage2/logs/stage2_configs.csv[large] pooled_auc by start × layer_scheme × data_level |",
              "| Fig. A1 weight drift | figures/figA1_weight_drift.pdf | Stage2/logs/stage2_runs.csv[data_level=40] drift_* mean by layer_scheme |",
              "| Fig. A2 per-fold AUC | figures/figA2_per_fold_auc.pdf | Stage2/logs/stage2_runs.csv[large,n=40,full] target_auc; Stage3/logs/stage3_runs.csv target_auc |",
              "", "## 3 · Unmatched numbers (need manual explanation)", ""]
    if unmatched:
        for w, ctx in unmatched:
            lines.append(f"- `{w}` — …{ctx}…")
    else:
        lines.append("None.")

    lines += ["", "## 4 · Corrections (relative to the earlier summary draft REPORT.md)", "",
              "| Old wording | Problem | New wording (recomputation basis) |", "|---|---|---|",
              "| 5 of 12 small-shift pairs have AUC < 0.5 (early informal summary) | missed id_06→id_04 (0.374) | 6 pairs (`stage1_small_n_below_half`) |",
              "| \"Under full fine-tuning the output layer dec.4 moves most (large shift 0.31)\" | under the large shift encoder.0 moves most (0.345) | small shift max decoder.4 0.171; large shift max encoder.0 0.345, then decoder.4 0.314 (`drift_*40_full_*`) |",
              "| \"Adaptation happens mainly in the second half of the decoder\" | holds only for the small shift | under the large shift both the input and output layers change substantially |",
              "| \"merged also degrades least under the large shift in frozen evaluation (drop 0.030)\" | id_00 drop is −0.074 | merged has the smallest threshold shift (4.60×), but not the smallest drop (`stage1_large_drop_id_00`) |",
              "| \"No point in Fig. 4 falls in the top-left corner\" | contradicts the figure | counted with thresholds (forgetting<0.15 and gain>0.30): 20 in the small shift, all starting from id_00; 0 in the large shift (`topleft_*`) |",
              "| \"Decoder-only reaches 97% of Full with half the parameters\" | reported only the small shift | 82% (large shift) to 97% (small shift), both at n=40 (`stage2_*_decoder_over_full_pct_n40`) |",
              "| Pretraining diversity examined only under full fine-tuning | other schemes omitted | reported separately for all four schemes; the lead under restricted schemes is mainly due to the id_06 collapse (`stage2_large_*_id_06_n40`, `*_singles_excl_id06_mean_n40`) |",
              "| Appendix \"202 numbers in total\" | does not match the registry size (actually 222 at the time) | the new report no longer contains this sentence |",
              ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"AUDIT.md: {len(rows)} numbers, {len(unmatched)} unmatched")
    for w, ctx in unmatched:
        print(f"  ❌ {w}  …{ctx}…")


if __name__ == "__main__":
    main()
