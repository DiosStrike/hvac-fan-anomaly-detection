# configs/

| File | Purpose |
| --- | --- |
| `config.yaml` | Single source of truth for all constants and rules; matches `proposal/proposal.md` v6 |
| `config.py` | Loads the yaml, runs startup self-checks, and holds the three pieces of deterministic logic that must be consistent across the whole project |

```bash
python code/configs/config.py   # print config summary + run self-checks
```

## What config.py provides

- `load()` — read the yaml, resolve active paths from `paths.env`, run self-checks
- `nested_order()` / `nested_levels()` — nested-sampling order, guaranteeing 5 ⊂ 10 ⊂ 20 ⊂ 40
- `loro_split()` / `target_finetune_sets()` — LORO fold split and fine-tuning samples per level
- `source_finetune_sets()` — small-shift ordering and level selection on the target ID's training pool
- `enumerate_stage2()` / `enumerate_stage3()` / `experiment_id()` — experiment enumeration and primary keys

Self-checks cover: training-pool arithmetic, merged-model per-ID equality and size alignment, matching
source/target class ratios, feature dimensions, LORO per-fold composition, validity of layer-scheme layer
names, and the 272/992/1032 counts. Any mismatch raises during `load()`, rather than running hundreds of
training runs on a wrong config.

## Implementation choices (not specified by the proposal)

1. **Filenames are the single source of truth for target-domain metadata.** `data/metadata/recording_log.csv` has 206 rows,
   including duplicate rows and misaligned-column records; after dedup, the 180 filenames form a complete 18×10 grid, so
   metadata is parsed from `{condition}_{voltage}_{noise}_run{NN}.wav` and the csv is only an inventory check.
2. **LORO fold k (0-based) holds out the whole run `run == k+1`.**
3. **Ordering = sort by filename, then shuffle once with `numpy.default_rng`.** Sorting first removes dependence on
   filesystem enumeration order; the RNG seed is derived with `crc32` rather than built-in `hash()`, which salts strings
   randomly and changes between processes. Seed scope: large shift includes the fold (each fold ordered separately), small shift includes the machine ID.
4. **"Bottleneck-adjacent layers only" = the two layers into and out of the bottleneck**: `encoder.4` (64→8) and `decoder.0` (8→64).
5. **`clip_score` = mean reconstruction MSE over all feature vectors of the clip**, aggregating each clip into one anomaly score.
6. **Fine-tuned weights are not saved** (`outputs.persist_finetuned_weights: false`); only the 5 pretrained weights +
   per-config evaluation CSVs are kept, avoiding a thousand-plus .pth files.

## Stage numbering (proposal v6; "unbiased source-model selection" removed, not a separate experiment)

0 pretraining and source baseline → 1 domain-shift quantification (frozen eval) → 2 main fine-tuning experiment (272 configs / 992 runs) → 3 no-pretraining control (4 configs / 40 runs)

## Fine-tuning hyperparameters (proposal §7, stage 2; finalized)

`finetune.lr = 0.0001` / `finetune.epochs = 30` / `finetune.batch_size = 64`,
shared by all four layer schemes, not tuned per scheme; no early stopping. Rationale in proposal/proposal.md, stage 2.
