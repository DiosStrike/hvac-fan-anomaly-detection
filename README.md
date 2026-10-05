# From Industrial to Household Fans: Quantifying Domain Shift and Limited-Data Fine-Tuning in Acoustic Anomaly Detection

Reconstruction-based acoustic anomaly detectors degrade when moved to new hardware. This project quantifies that degradation and how to repair it cheaply. Five autoencoders are pretrained on MIMII fan sounds (four single-ID models and one mixed-ID model). They are then evaluated under two shifts: a small shift to an unseen MIMII machine ID, and a large shift to a self-recorded consumer household fan. The models are fine-tuned with 5/10/20/40 normal target clips under four layer schemes (full, decoder-only, encoder-only, bottleneck-only), with leave-one-round-out cross-validation for the large shift. In total there are 272 fine-tuning configurations (992 runs) and a 40-run no-pretraining control. Main findings:

- The threshold shift and the AUC drop disagree.
- About 20 target clips suffice for the large shift.
- Tuning the decoder recovers most of the gain at about half the parameter cost.
- Pretraining helps most in the 20-clip regime.

This is an **independent research project**. See the full write-up in **[report/report.pdf](report/report.pdf)**.

## Repository structure

```
proposal/proposal.md          Research proposal and locked experimental protocol
code/
  configs/                    config.yaml (all constants), config.py (loader, self-checks, sampling/LORO logic)
  notebooks/                  Experiment notebooks (Google Colab)
    baseline_runner.ipynb       MIMII baseline autoencoder, per machine ID
    stage0_pretrain.ipynb       Stage 0: pretraining + source-domain baselines (5 models)
    stage1_domain_shift.ipynb   Stage 1: frozen evaluation under small / large shift
    stage2_finetune.ipynb       Stage 2: fine-tuning grid (272 configs / 992 runs)
    stage3_no_pretrain.ipynb    Stage 3: no-pretraining control (4 configs / 40 runs)
  analysis/                   Scripts that recompute report tables/numbers (and figures) from results/
data/
  raw_audio/                  Self-recorded household-fan audio (180 WAV clips)
  metadata/recording_log.csv  Recording log
  photos/fan_conditions.png   Photos of the fan in its three operating states
results/                      Metrics and logs from all stages (CSV / JSON / JSONL)
report/report.pdf             Final report
requirements.txt
```

## How to run

The experiment notebooks were written for **Google Colab** (GPU runtime) with data on Google Drive.

1. Download `0_dB_fan.zip` from MIMII (see below). Zip the self-recorded audio from the repo root with `cd data && zip -r raw_audio.zip raw_audio`.
2. Put both zips in `MyDrive/HVAC_fan/` on Google Drive: `0_dB_fan.zip` and `raw_audio.zip`. These paths are set in `code/configs/config.yaml` (`paths.colab`) and in the edit block at the top of each stage notebook.
3. Run the notebooks in order: `stage0_pretrain` → `stage1_domain_shift` → `stage2_finetune` → `stage3_no_pretrain`. Each stage writes to `MyDrive/HVAC_fan/HVAC2_Results/<Stage>/{logs,models}` and reads the previous stage's outputs from there. Stage 2 and Stage 3 resume automatically if a session is interrupted.
4. To regenerate the report's tables, figures and number registry from the archived results, run this locally from the repo root:

   ```bash
   pip install -r requirements.txt
   python code/configs/config.py                    # print the config summary and run self-checks
   python code/analysis/make_report_figures.py      # reads results/, writes report_figures/ and report/{figures,tables}
   ```

   `code/analysis/audit_report.py` checks every number in the report's LaTeX source against the registry. The LaTeX source is not included in this repository.

## Data

**Source domain: MIMII (not included).** Download `0_dB_fan.zip` from the MIMII dataset on Zenodo: <https://zenodo.org/records/3384388> (Purohit et al., 2019).
- For Colab runs, place the zip at `MyDrive/HVAC_fan/0_dB_fan.zip`.
- For local runs (`paths.env: local` in `config.yaml`), extract it so that the machine IDs sit at `data/mimii/fan/id_00`, `data/mimii/fan/id_02`, `data/mimii/fan/id_04` and `data/mimii/fan/id_06`, each with `normal/` and `abnormal/`.

**Target domain: self-recorded household fan (included).** There are 180 clips in `data/raw_audio/{condition}/{voltage}/{environment}/`, each 10 s, 16 kHz, mono, recorded with a single USB microphone (Razer Seiren V3 Mini). The full design is 3 conditions (normal / blocked / imbalance) × 3 supply voltages (4 V / 8 V / 12 V) × 2 environments (quiet / noise) × 10 recording rounds. File names encode the metadata (`{condition}_{voltage}_{environment}_run{NN}.wav`). `data/metadata/recording_log.csv` is the original recording log; it contains duplicate rows and is used only as a manifest. `data/photos/fan_conditions.png` shows the fan in the three conditions.

**Not included in this repository:**

| What | Why | How to get it |
|---|---|---|
| MIMII `0_dB_fan.zip` | Public dataset | Download from Zenodo (link above) |
| Model weights (`*.pth`: 5 pretrained models, Stage 2/3 checkpoints, ~64 MB) | Weights are not distributed | Re-run `stage0_pretrain` (pretrained models), then `stage2_finetune` / `stage3_no_pretrain` |
| Stage 2 per-clip anomaly scores (`results/Stage2/logs/clip_scores/`, 272 files, ~64 MB) | Size | Written by `stage2_finetune.ipynb`. The aggregated metrics are in `results/Stage2/logs/`. |
| Cached features (`.npy`) and generated figures | Derived files | Recreated by the notebooks / analysis scripts |
