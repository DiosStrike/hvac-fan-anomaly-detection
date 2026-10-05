"""HVAC 2.0 config loading and deterministic split rules.

This module is the single entry point to configs/config.yaml, and additionally provides three things
that must be "consistent across the whole project":

  1. Derived constants and self-checks  -- verify at startup that the numbers in the yaml are mutually consistent (272 / 992 / 1032 etc.)
  2. Nested-sampling order              -- 5 ⊂ 10 ⊂ 20 ⊂ 40 (proposal §4.3)
  3. LORO fold split                    -- hold out a whole run by run number (proposal §6.1)

No training logic here; depends only on numpy + pyyaml.
Running `python code/configs/config.py` directly prints a config summary and runs the self-checks.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the yaml, resolve active paths from paths.env, run self-checks, return a dict."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["paths"]["resolved"] = _resolve_paths(cfg["paths"])
    self_check(cfg)
    return cfg


def _resolve_paths(paths: dict[str, Any]) -> dict[str, str | None]:
    env = paths["env"]
    if env not in ("colab", "local"):
        raise ValueError(f"paths.env must be colab or local, got {env!r}")
    out: dict[str, str | None] = {}
    for key, value in paths[env].items():
        out[key] = str(Path(value).expanduser()) if value else None
    return out


def out_dir(cfg: dict[str, Any], which: str) -> Path:
    """Join outputs.{which}_dir onto outputs_root. which ∈ {pretrained_models, results, figures}"""
    root = Path(cfg["paths"]["resolved"]["outputs_root"])
    return root / cfg["outputs"][f"{which}_dir"]


# ---------------------------------------------------------------------------
# Self-check: numbers in the yaml must be mutually consistent; better to fail at startup
# ---------------------------------------------------------------------------


def self_check(cfg: dict[str, Any]) -> None:
    src, tgt, grid, stages = (
        cfg["source_domain"],
        cfg["target_domain"],
        cfg["grid"],
        cfg["stages"],
    )

    # --- Source domain: training pool = total normals − normals used by the eval set ---
    n_eval_normal = src["eval_set"]["n_normal"]
    n_eval_abnormal = src["eval_set"]["n_abnormal"]
    for mid in src["machine_ids"]:
        counts = src["official_counts"][mid]
        expect = counts["normal"] - n_eval_normal
        got = src["train_pool_size"][mid]
        assert got == expect, f"{mid} training pool should be {expect}, yaml says {got}"
        assert counts["abnormal"] >= n_eval_abnormal, (
            f"{mid} has only {counts['abnormal']} abnormal clips, cannot draw {n_eval_abnormal} eval samples"
        )

    # --- Merged model: equal per ID, and total within the single-ID range ---
    merged, n_ids = src["merged_model"], len(src["machine_ids"])
    assert merged["eval_per_id"]["normal"] * n_ids == n_eval_normal
    assert merged["eval_per_id"]["abnormal"] * n_ids == n_eval_abnormal
    assert merged["train_per_id"] * n_ids == merged["train_total"]
    pools = list(src["train_pool_size"].values())
    assert min(pools) <= merged["train_total"] <= max(pools), (
        f"merged model training size {merged['train_total']} is outside the single-ID range "
        f"[{min(pools)}, {max(pools)}]; 'data volume' and 'source diversity' would be confounded"
    )
    assert merged["train_per_id"] <= min(pools), "merged model per-ID sample count exceeds that ID's training pool"

    # --- Eval-set class ratio: source and target must match, else best-F1 is not comparable across shifts ---
    src_ratio = n_eval_abnormal / n_eval_normal
    tgt_ratio = tgt["n_abnormal_total"] / tgt["n_normal_total"]
    assert abs(src_ratio - tgt_ratio) < 1e-9, (
        f"source ratio 1:{src_ratio} differs from target 1:{tgt_ratio}; best-F1 trivial floors would differ"
    )
    floor = 2 * src_ratio / (src_ratio + 1) / (1 + src_ratio / (src_ratio + 1))
    assert abs(floor - cfg["metrics"]["best_f1_trivial_floor"]) < 1e-9, (
        f"best_f1_trivial_floor should be {floor:.3f}"
    )

    # --- Target domain: complete 18 × 10 grid ---
    n_cells = (
        len(tgt["conditions"]) * len(tgt["voltages"]) * len(tgt["noise_levels"])
    )
    assert n_cells * tgt["n_runs"] == tgt["n_clips_total"] == 180
    n_normal_cells = (
        len(tgt["normal_conditions"]) * len(tgt["voltages"]) * len(tgt["noise_levels"])
    )
    assert n_normal_cells * tgt["n_runs"] == tgt["n_normal_total"]

    # --- Feature dimensions ---
    feat = cfg["features"]
    assert feat["n_mels"] * feat["frames"] == feat["input_dim"]
    assert cfg["model"]["encoder_dims"][0] == feat["input_dim"]
    assert cfg["model"]["decoder_dims"][-1] == feat["input_dim"]
    assert cfg["model"]["encoder_dims"][-1] == cfg["model"]["bottleneck_dim"]

    # --- LORO: leave one run out → folds = runs, per-fold composition = 18 = 6 + 12 ---
    loro = cfg["loro"]
    assert loro["n_folds"] == tgt["n_runs"]
    assert loro["clips_per_fold"] == n_cells
    assert loro["normals_per_fold"] == n_normal_cells
    assert loro["normals_per_fold"] + loro["abnormals_per_fold"] == loro["clips_per_fold"]
    assert loro["max_train_normals"] == tgt["n_normal_total"] - loro["normals_per_fold"]
    assert max(grid["data_levels"]) <= loro["max_train_normals"], (
        "data level exceeds the number of normals available under LORO"
    )

    # --- Nested sampling: levels must be strictly increasing, otherwise "containment" is meaningless ---
    levels = grid["data_levels"]
    assert levels == sorted(set(levels)), "data_levels must be strictly increasing with no duplicates"

    # --- Layer schemes: every trainable layer name must be a real layer ---
    known = set(cfg["model"]["layer_index"]["encoder_linears"]) | set(
        cfg["model"]["layer_index"]["decoder_linears"]
    )
    for name, scheme in grid["layer_schemes"].items():
        unknown = set(scheme["trainable"]) - known
        assert not unknown, f"layer scheme {name} references nonexistent layers {sorted(unknown)}"
        assert scheme["trainable"], f"layer scheme {name} has no trainable layers"
    assert set(grid["layer_schemes"]["full"]["trainable"]) == known, "full should cover all layers"

    # --- Experiment counts: must equal what the grid actually enumerates ---
    n_levels, n_schemes = len(levels), len(grid["layer_schemes"])
    small, large = stages["stage2_finetune"]["small_shift"], stages["stage2_finetune"]["large_shift"]
    assert small["n_configs"] == small["starts"] * small["targets_per_start"] * n_levels * n_schemes
    assert large["n_configs"] == large["starts"] * large["targets_per_start"] * n_levels * n_schemes
    assert small["n_runs"] == small["n_configs"] * small["folds"]
    assert large["n_runs"] == large["n_configs"] * large["folds"]
    assert stages["stage2_finetune"]["n_configs_total"] == small["n_configs"] + large["n_configs"]
    assert stages["stage2_finetune"]["n_runs_total"] == small["n_runs"] + large["n_runs"]

    s3 = stages["stage3_no_pretrain"]
    assert s3["n_configs"] == n_levels
    assert s3["n_runs"] == s3["n_configs"] * s3["folds"]
    assert cfg["total_runs"] == stages["stage2_finetune"]["n_runs_total"] + s3["n_runs"]

    # --- Stage 1: small shift is "each start × the other IDs"; large shift includes the merged model ---
    s1 = stages["stage1_frozen_eval"]
    assert s1["small_shift_pairs"] == len(src["machine_ids"]) * (len(src["machine_ids"]) - 1)
    assert s1["large_shift_models"] == stages["stage0_pretrain"]["n_models"]
    assert s1["large_shift_folds"] == loro["n_folds"]

    # Number of starts matches the models produced in stage 0: small shift uses single-ID only, large shift uses all
    assert small["starts"] == len(src["machine_ids"])
    assert large["starts"] == stages["stage0_pretrain"]["n_models"]


# ---------------------------------------------------------------------------
# Deterministic RNG: seed derived from seed + several strings/ints, stable across processes and machines
# ---------------------------------------------------------------------------


def rng_for(seed: int, *parts: Any) -> np.random.Generator:
    """Derive a stable Generator from (seed, parts...).

    Uses crc32 rather than built-in hash() -- the latter salts str randomly and changes between processes.
    """
    tag = "|".join(str(p) for p in parts).encode("utf-8")
    return np.random.default_rng([seed, zlib.crc32(tag)])


# ---------------------------------------------------------------------------
# Nested sampling: 5 ⊂ 10 ⊂ 20 ⊂ 40 (proposal §4.3)
# ---------------------------------------------------------------------------


def nested_order(items: Iterable[str], seed: int, *scope: Any) -> list[str]:
    """Put candidate samples into a fixed order: sort by filename first (removing dependence on
    filesystem enumeration order), then shuffle once with an RNG derived from (seed, scope).

    Level n takes the first n, so levels naturally satisfy containment.
    """
    ordered = sorted(items)
    idx = rng_for(seed, *scope).permutation(len(ordered))
    return [ordered[i] for i in idx]


def take_level(ordered: Sequence[str], n: int) -> list[str]:
    """Take level n from an ordered list."""
    if n > len(ordered):
        raise ValueError(f"level {n} exceeds the number of available samples {len(ordered)}")
    return list(ordered[:n])


def nested_levels(items: Iterable[str], levels: Sequence[int], seed: int, *scope: Any):
    """Return all levels at once and assert that containment actually holds."""
    ordered = nested_order(items, seed, *scope)
    out = {n: take_level(ordered, n) for n in levels}
    for smaller, larger in zip(levels, levels[1:]):
        assert out[smaller] == out[larger][:smaller], "nesting is broken"
    return out


# ---------------------------------------------------------------------------
# Target domain: filename parsing and LORO fold split (proposal §6.1)
# ---------------------------------------------------------------------------

_CLIP_RE = re.compile(r"^(?P<condition>\w+?)_(?P<voltage>\d+V)_(?P<noise>\w+?)_run(?P<run>\d{2})\.wav$")


def parse_clip(filename: str) -> dict[str, Any]:
    """Parse metadata from a filename. The filename is the single source of truth; recording_log.csv is not used."""
    m = _CLIP_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"filename does not follow the naming convention: {filename}")
    d = m.groupdict()
    d["run"] = int(d["run"])
    return d


def clip_relpath(cfg: dict[str, Any], filename: str) -> str:
    meta = parse_clip(filename)
    return cfg["target_domain"]["relpath_pattern"].format(filename=Path(filename).name, **meta)


def is_normal(cfg: dict[str, Any], filename: str) -> bool:
    return parse_clip(filename)["condition"] in cfg["target_domain"]["normal_conditions"]


def loro_split(cfg: dict[str, Any], filenames: Sequence[str], fold: int):
    """Fold `fold` (0-based): hold out the whole run == fold+1 as the test set.

    Returns (train_normals, test_clips). Note the training side contains only normals -- the autoencoder
    is never trained on abnormal clips; the 108 abnormal recordings of non-test runs never enter training at any stage.
    """
    n_folds = cfg["loro"]["n_folds"]
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    held_out_run = fold + 1

    test_clips, train_normals = [], []
    for f in filenames:
        meta = parse_clip(f)
        if meta["run"] == held_out_run:
            test_clips.append(f)
        elif meta["condition"] in cfg["target_domain"]["normal_conditions"]:
            train_normals.append(f)

    loro = cfg["loro"]
    assert len(test_clips) == loro["clips_per_fold"], (
        f"fold {fold} test set has {len(test_clips)} clips, expected {loro['clips_per_fold']}"
    )
    assert len(train_normals) == loro["max_train_normals"], (
        f"fold {fold} training side has {len(train_normals)} normal clips, expected {loro['max_train_normals']}"
    )
    return sorted(train_normals), sorted(test_clips)


def target_finetune_sets(cfg: dict[str, Any], filenames: Sequence[str], fold: int):
    """Fine-tuning samples for each data level under fold `fold` (large shift).

    Each fold is ordered separately -- because the 54 available normals differ per fold (proposal §4.3).
    """
    train_normals, _ = loro_split(cfg, filenames, fold)
    scope = cfg["grid"]["nested_sampling"]["large_shift_seed_scope"]
    parts = [fold if s == "fold" else s for s in scope if s != "seed"]
    return nested_levels(train_normals, cfg["grid"]["data_levels"], cfg["seed"], "large_shift", *parts)


def source_finetune_sets(cfg: dict[str, Any], train_pool: Sequence[str], target_id: str):
    """Small shift: order the target ID's training pool and take levels.

    train_pool must be that ID's normals after removing the eval set, and must never contain any eval-set clip
    (sampling-pool constraint of proposal §4.1).
    """
    return nested_levels(train_pool, cfg["grid"]["data_levels"], cfg["seed"], "small_shift", target_id)


# ---------------------------------------------------------------------------
# Experiment enumeration
# ---------------------------------------------------------------------------


def enumerate_stage2(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The 272 stage-2 experiment configs (LORO folds not expanded)."""
    ids = cfg["source_domain"]["machine_ids"]
    levels = cfg["grid"]["data_levels"]
    schemes = list(cfg["grid"]["layer_schemes"])
    configs = []

    for start in ids:  # small shift: 4 starts × 3 unseen IDs each
        for target in [t for t in ids if t != start]:
            for n in levels:
                for scheme in schemes:
                    configs.append(
                        dict(stage="stage2", shift="small", start=start, target=target,
                             data_level=n, layer_scheme=scheme, n_folds=1, source_eval=start)
                    )

    for start in ids + ["merged"]:  # large shift: 5 starts → self-recorded fan
        for n in levels:
            for scheme in schemes:
                configs.append(
                    dict(stage="stage2", shift="large", start=start, target="self_recorded_fan",
                         data_level=n, layer_scheme=scheme, n_folds=cfg["loro"]["n_folds"],
                         source_eval=start)
                )
    return configs


def enumerate_stage3(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The 4 stage-3 no-pretraining control configs."""
    s3 = cfg["stages"]["stage3_no_pretrain"]
    return [
        dict(stage="stage3", shift="large", start="random_init", target="self_recorded_fan",
             data_level=n, layer_scheme=s3["layer_scheme"], n_folds=s3["folds"], source_eval=None)
        for n in cfg["grid"]["data_levels"]
    ]


def experiment_id(c: dict[str, Any]) -> str:
    """Stable identifier for a config, used as the CSV primary key and log prefix."""
    return f"{c['stage']}_{c['shift']}_{c['start']}_to_{c['target']}_n{c['data_level']}_{c['layer_scheme']}"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summary(cfg: dict[str, Any]) -> str:
    s2, s3 = enumerate_stage2(cfg), enumerate_stage3(cfg)
    small = [c for c in s2 if c["shift"] == "small"]
    large = [c for c in s2 if c["shift"] == "large"]
    runs = sum(c["n_folds"] for c in s2 + s3)
    src = cfg["source_domain"]
    lines = [
        f"{cfg['project']} (proposal {cfg['proposal_version']}), seed={cfg['seed']}, env={cfg['paths']['env']}",
        f"  source   {src['name']}  {', '.join(src['machine_ids'])}",
        f"           eval set {src['eval_set']['n_normal']} normal + {src['eval_set']['n_abnormal']} abnormal"
        f"  train pool {min(src['train_pool_size'].values())}~{max(src['train_pool_size'].values())}",
        f"           merged model {src['merged_model']['train_per_id']} per ID = {src['merged_model']['train_total']}",
        f"  target   {cfg['target_domain']['n_clips_total']} clips"
        f" ({cfg['target_domain']['n_normal_total']} normal / {cfg['target_domain']['n_abnormal_total']} abnormal)"
        f"  LORO {cfg['loro']['n_folds']} folds",
        f"  grid     data levels {cfg['grid']['data_levels']} (nested) × layer schemes {list(cfg['grid']['layer_schemes'])}",
        f"  stage 2  small shift {len(small)} configs / {sum(c['n_folds'] for c in small)} runs"
        f"  | large shift {len(large)} configs / {sum(c['n_folds'] for c in large)} runs",
        f"  stage 3  {len(s3)} configs / {sum(c['n_folds'] for c in s3)} runs",
        f"  total    {len(s2) + len(s3)} configs / {runs} train/eval runs",
        f"  metrics  primary {cfg['metrics']['primary']} | secondary {cfg['metrics']['secondary']}"
        f" (trivial floor {cfg['metrics']['best_f1_trivial_floor']})",
        f"  fine-tune hparams lr={cfg['finetune']['lr']} epochs={cfg['finetune']['epochs']} "
        f"batch_size={cfg['finetune']['batch_size']}"
        f" (early stopping={'on' if cfg['finetune']['early_stopping'] else 'off'})",
    ]
    assert runs == cfg["total_runs"], f"enumerated {runs} runs, yaml declares {cfg['total_runs']}"
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = load()
    print(summary(cfg))
    print("\nSelf-check passed.")
