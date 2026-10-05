# Domain Adaptation for Acoustic Anomaly Detection in Low-Resource Cross-Hardware Settings

## 1. Background and Motivation

Pretrained acoustic anomaly detection models suffer from severe domain shift when deployed across hardware. A model trained on industrial-grade equipment (a research-grade 8-channel microphone array in a controlled laboratory environment) and deployed directly on consumer-grade hardware (a single-channel USB microphone in a real household environment) can see its alarm threshold shift by a factor of 5 to 10, rendering the model unusable.

This study focuses on three core topics: **domain shift**, **limited-data adaptation**, and **fine-tuning strategy**.

## 2. Core Research Questions

This study answers only three questions; all other topics are out of scope:

- **RQ1 (Shift magnitude)**: How does performance degradation differ fundamentally between a small shift (switching between industrial machine models) and a large shift (industrial-grade to consumer-grade hardware)?
- **RQ2 (Data efficiency)**: How much target-domain data is needed for effective adaptation? Does this requirement change with shift magnitude?
- **RQ3 (Fine-tuning strategy)**: Which layers are most effective to adapt? How should "target-domain gain" be traded off against "source-domain forgetting"? Does the hardware diversity seen during pretraining reduce the data required for subsequent adaptation?

## 3. Evaluation Metrics

All stages report the same two metrics:

- **AUC (primary metric)**: measures the model's ability to rank anomalous samples above normal samples.
- **Best F1 (secondary metric)**: sweep every possible decision threshold and take the maximum F1 score.

Both are **threshold-independent metrics**: they reflect only the model's ranking quality and do not depend on any particular threshold setting. This matches the scope of the study: the object of study is "how to adapt the model", not "how to set the deployment threshold". Threshold calibration is discussed under future work in Section 10.

**On the trivial floor of Best F1:** Best F1 has a floor determined by the normal-to-anomalous ratio of the evaluation set. The threshold "flag every sample as anomalous" always lies within the sweep range, so a model with no discriminative power at all still attains this floor score. If the source and target evaluation sets have different normal-to-anomalous ratios, their F1 starting points differ and cross-domain comparison breaks down.

This study eliminates the problem by **fixing the normal-to-anomalous ratio of both the source and target evaluation sets at 1:2**:

| Evaluation set | Normal : Anomalous | Trivial floor of Best F1 |
| --- | --- | --- |
| Small shift (MIMII) | 160 : 320 | 0.800 |
| Large shift (self-recorded fan) | 60 : 120 | 0.800 |

The two floors are identical, so **Best F1 can be compared across shift magnitudes**.

Even so, AUC remains the primary metric: its floor is always 0.5 and is completely independent of the normal-to-anomalous ratio, so it does not rely on the ratio convention above and is more robust. Best F1 serves as a secondary metric that offers a view closer to actual decision performance.

**Note:** This ratio deliberately departs from the convention in `code/notebooks/baseline_runner.ipynb`, which pairs the anomalous samples with an equal number of normal samples (320:320). The change is made to align with the target domain's natural 1:2 ratio (60 normal vs. 120 anomalous clips); keep this difference in mind when comparing against that notebook.

## 4. Data

### 4.1 Source-Domain Data (MIMII public dataset, 0_dB_fan)

We use four machine IDs from the **0 dB SNR** version of the MIMII fan sound dataset: id_00, id_02, id_04, and id_06. The official normal/anomalous sample counts are:

| Machine ID | Normal samples | Anomalous samples |
| --- | --- | --- |
| id_00 | 1011 | 407 |
| id_02 | 1016 | 359 |
| id_04 | 1033 | 348 |
| id_06 | 1015 | 361 |

Evaluation-set construction follows the overall procedure of `code/notebooks/baseline_runner.ipynb` (a mixed normal + anomalous evaluation set, with the evaluation set drawn first), but **the normal-to-anomalous ratio is changed to 1:2** to align with the target domain: every model's evaluation set is fixed at **160 normal + 320 anomalous = 480 samples**. The rationale is given in Section 3. The exact construction of training and evaluation sets differs between single-ID models and the merged model:

**Single-ID models (4):** Each model uses only data from its own machine ID.

- Evaluation set: **160 normal + 320 anomalous samples** drawn from that ID
- Training set: all normal samples of that ID minus the 160 normal samples used by the evaluation set

The four IDs have 407/359/348/361 anomalous samples respectively, all above 320, so drawing 320 anomalous samples is not a problem (id_04 is the tightest, with 28 to spare).

Because the four IDs already differ in their total number of normal samples (1011 to 1033), the resulting training-set sizes are not exactly equal:

| Machine ID | Total normal samples | Minus 160 used for evaluation | Training pool size |
| --- | --- | --- | --- |
| id_00 | 1011 | −160 | 851 |
| id_02 | 1016 | −160 | 856 |
| id_04 | 1033 | −160 | 873 |
| id_06 | 1015 | −160 | 855 |

This difference is not forcibly equalized. It is kept both in this proposal and in the code, and recorded as an inherent minor difference in data volume among the four single-ID models.

**Merged model (1):** Both training and evaluation data are drawn evenly from the four IDs so that each ID contributes equally. The sampling order is the same as for the single-ID models, with the evaluation set drawn first.

- Evaluation set: **40 normal + 80 anomalous** per ID, giving 160 normal + 320 anomalous = 480 samples across the four IDs, exactly matching the size and ratio of the single-ID evaluation sets. **These samples must be drawn from that ID's already-fixed evaluation subset (160 normal / 320 anomalous) and must not be resampled independently**, so that "the evaluation set is always the evaluation set" and no training sample of an ID can leak into the merged model's evaluation set.
- Training set: after removing the normal samples used by the evaluation set, **215** samples are drawn at random from each ID's remaining normal samples, for a total of **860** across the four IDs; the other remaining samples are not used for training.

Unlike the single-ID models, where "all remaining samples are used for training", the merged model's training set undergoes an additional fixed-size subsampling step (215 per ID) after the evaluation set is drawn. The reason is that the four IDs together have roughly 3,400 remaining samples; using all of them would give the merged model about four times as much data as a single-ID model, confounding "more data" with "more diverse sources". Capping it at 860 makes the merged model's total training volume roughly equal to that of the single-ID models (851 to 873), so the only difference is whether the data come from a single ID or a mixture of four.

**Strict sampling-pool constraint (written into `code/configs/config.yaml`; must not be violated):** For small-shift fine-tuning, the fine-tuning samples for a target ID **may only be drawn from that ID's training pool** (id_00: 851 / id_02: 856 / id_04: 873 / id_06: 855) and **must never touch that ID's evaluation set**, because the 160 normal + 320 anomalous evaluation set is exactly what is used to measure adaptation. All sampling uses the fixed random seed **seed = 42**.

### 4.2 Target-Domain Data (self-recorded fan)

180 recordings (stored in `data/raw_audio/`, logged in `data/metadata/recording_log.csv`): 3 states (normal / blocked / imbalance) × 3 voltage levels (4V / 8V / 12V) × 2 environments (quiet / noisy) × 10 repetitions each, 10 s / 16 kHz / mono. Of these:

- **60 normal clips**: used for fine-tuning, with data-volume levels set to **5 / 10 / 20 / 40** (4 levels). The levels stop at 40 for the reason given in Section 6: under the LORO protocol, each held-out test round occupies 6 normal clips, so at most 54 normal clips are actually available.
- **120 blocked + imbalance clips**: used only for evaluation and never for training. Because the autoencoder is trained to "reconstruct normal sounds", anomalous samples never enter training at any stage; these 120 of the 180 clips appear only at test time. This is an inherent property of the autoencoder anomaly-detection paradigm, not a waste of data.

Small-shift fine-tuning (between MIMII IDs) uses the same 5/10/20/40 data-volume levels as the large shift, so that "different data volume" does not interfere with the variable "different shift magnitude".

### 4.3 Nested Sampling of Data-Volume Levels (important)

The four data-volume levels **must use nested sampling**, i.e., 5 ⊂ 10 ⊂ 20 ⊂ 40:

1. First shuffle the available training samples with seed = 42 to fix a single ordering.
2. The 5-clip level takes the first 5, the 10-clip level the first 10, the 20-clip level the first 20, and the 40-clip level the first 40.

This gives a strict inclusion relationship between levels: going from 5 to 10 clips, **the original 5 are left completely untouched and the only change is the 5 added clips**. If each level were sampled independently at random, the conclusion "40 clips beats 5 clips" would be contaminated by the luck of the 40-clip draw happening to be more representative, making it impossible to tell whether the improvement comes from data volume or sampling luck.

Scope: for the large shift, each fold defines its own ordering (since the available samples differ per fold); the same applies to each ID's training pool for the small shift.

## 5. Model and Pretraining

The architecture, feature extraction, and training recipe follow `code/notebooks/baseline_runner.ipynb`: 320-dimensional input (64 mel bins × 5 concatenated frames) → 64 → 64 → **8-dimensional bottleneck** → 64 → 64 → 320-dimensional output, ReLU activations, Adam optimizer, MSE loss, batch size 512, 50 epochs.

**Pretraining produces 5 models:**

- 4 single-ID models: id_00 / id_02 / id_04 / id_06, each trained independently on all normal samples remaining after the evaluation set is removed (851 to 873 samples; see Section 4.1)
- 1 merged-ID model: 215 samples from each of the four IDs, 860 in total, trained as a mixture. Its total training volume is matched to the single-ID models, and it serves as the "more hardware diversity seen during pretraining" comparison.

## 6. Validation Protocol (LORO)

### 6.1 Large Shift (self-recorded fan): Leave-One-Round-Out Cross-Validation

The 180 recordings are organized as follows: each of the 18 condition combinations (3 states × 3 voltages × 2 environments) was recorded 10 times, and each "round" contains one clip of every one of the 18 conditions, giving 10 rounds × 18 clips = 180 clips. The 10 repetitions of the same condition were captured consecutively on the same device, at the same position, within a short time, and are highly similar to one another. With an ordinary random split, different repetitions of the same condition would end up on both sides of the train/test split; the model would already have seen "close relatives" of the test samples during training, and the resulting metrics would reflect memorization rather than generalization.

**Each LORO fold holds out an entire round of 18 clips as the test set; training never touches any data from that round. There are 10 folds in total.**

**Key implementation requirement: every fold must restart fine-tuning from the pretrained model and must not continue training from the previous fold's model.** Reason: fold 1's training data include samples from round 2, and round 2 is exactly fold 2's test set; carrying the model over would constitute leakage. Hence one experimental configuration under LORO produces 10 mutually independent fine-tuned models.

### 6.2 Two Aggregation Schemes for Metrics

After all 10 folds have run, both schemes are computed and reported:

- **Pooled scheme**: collect the anomaly scores of the test samples from all 10 folds (10 folds × 18 clips = 180 clips, each tested exactly once) and compute a single AUC and a single Best F1. The sample size increases from 18 per fold to 180, which is statistically more stable.
- **Per-fold scheme**: compute AUC and Best F1 separately for each fold, yielding 10 sets of numbers, and report their **mean and standard deviation**. Each fold contains only 6 normal + 12 anomalous samples, so per-fold metrics are coarse and high-variance, but the size of their fluctuation indicates how reliable the result is.

The difference between the two schemes is itself diagnostic: if they are close, the result is robust; if the pooled value is clearly below the per-fold mean, this suggests cross-fold scale drift (see Limitations).

### 6.3 Constraint on Data-Volume Levels

Each held-out test round contains 6 normal clips (one for each of the 6 normal conditions), so the upper limit of normal clips actually available for training is 60 − 6 = **54**. The fine-tuning data-volume levels are therefore set to 5/10/20/40, with no 60-clip level.

### 6.4 Small Shift (between MIMII IDs)

This follows the official MIMII train/test split logic. Evaluation uses the **target ID's own evaluation set of 160 normal + 320 anomalous samples**. Each MIMII ID has over a thousand recordings and lacks the strongly correlated "same condition recorded consecutively" structure, so no LORO treatment is needed.

## 7. Experimental Design

### Stage 0: Pretraining and Source-Domain Baselines

See Section 5; this stage produces 5 pretrained models.

**Immediately after each model finishes training, evaluate it once on its own 480-sample source-domain evaluation set and record AUC and Best F1 as the source-domain baseline.** These 5 numbers are the reference against which "forgetting" is later measured: Stage 2 reports post-fine-tuning source-domain performance, but "how much was forgotten" is only meaningful relative to the pre-fine-tuning level.

The source-domain evaluation target after fine-tuning is defined as follows:

- Single-ID starting point (whether adapted to the small or the large shift) → its **own ID's** 160 normal + 320 anomalous
- Merged-model starting point → the merged 160 normal + 320 anomalous (40 normal + 80 anomalous from each of the four IDs)

### Stage 1: Quantifying Domain Shift (frozen evaluation, no training)

**All 5 pretrained models take part in frozen evaluation** (4 single-ID + 1 merged):

- **Small shift**: only the 4 single-ID models take part, each evaluated on the 3 MIMII IDs it has not seen (3 pairs per model, 12 pairs in total), using the target ID's 160 normal + 320 anomalous evaluation set.
- **Large shift**: **all 5 models are evaluated** on the self-recorded fan recordings. The merged model skips the small shift (it has seen all four IDs during pretraining) but must be evaluated on the large shift. Otherwise the gain attributable to the "pretraining diversity effect" in RQ3 cannot be computed: with only post-fine-tuning numbers and no pre-fine-tuning starting point, it is impossible to tell whether the benefit of diversity shows up as "a higher starting point" or as "more data-efficient fine-tuning", which are two entirely different conclusions.

**Frozen evaluation on the large shift must use exactly the same 10-fold LORO split as Stage 2**: each fold evaluates only that round's 18 test clips, and results are likewise aggregated with both the pooled and per-fold schemes. Although frozen evaluation involves no training and carries no risk of leakage, only an identical data split allows the Stage 1 baseline numbers to be subtracted directly from the Stage 2 post-fine-tuning numbers to obtain a strictly comparable "gain".

Recorded metrics: AUC, Best F1, median reconstruction MSE on normal samples, and threshold-shift factor.

**Definition of the threshold-shift factor:**

```
shift factor = median reconstruction MSE of target-domain normal samples ÷ median reconstruction MSE of source-domain normal samples
```

The sample ranges for the numerator and denominator are defined explicitly:

- **Denominator (source domain)**: the **160 normal samples in that ID's evaluation set**, not training-pool samples. The model has already fit the training-pool samples, so using them as the denominator would deflate the baseline and inflate the shift factor.
- **Numerator (target domain)**:
  - Small shift → the 160 normal samples in the target ID's evaluation set
  - Large shift → **all 60 normal recordings**. This is not computed within LORO folds, because Stage 1 is a frozen evaluation in which the model is not trained at all, so there is no risk of data leakage; a median over 60 clips is far more stable than one over the 6 clips in a single fold.

The "median reconstruction MSE on normal samples" recorded in this section is exactly the numerator and denominator above; both uses share the same pair of numbers, and no separate aggregation is defined.

The median is chosen over the mean or p95 because it is robust, unaffected by a few extreme recordings, and semantically clear: it answers "by how many times is the reconstruction error of a typical normal sound amplified?". This metric directly quantifies the "5 to 10× threshold shift" phenomenon described at the start of this proposal.

### Stage 2: Main Fine-Tuning Experiments (core workload)

**Variable grid:**

- Data volume: 5 / 10 / 20 / 40 (4 levels, nested sampling; see Section 4.3)

- Layer-adaptation strategy: full fine-tuning / decoder only / encoder only / near-bottleneck layers only (4 strategies)

- Learning rate, number of epochs, and fine-tuning batch size: fixed across all experiments and not treated as comparison variables. **These are set to lr = 1e-4 / epochs = 30 / batch_size = 64**:
  - **batch_size = 64**: The pretraining value of 512 is too large for limited-data fine-tuning. Five clips yield about 1,545 feature vectors; at batch size 512 that is only 4 batches per epoch, or 120 gradient updates over 30 epochs. Adam's moment estimates (β₂ = 0.999, which needs roughly a thousand steps to warm up) would have no time to warm up, which is equivalent to not training at all. At 64, the 5-clip level gets about 750 updates, a normal order of magnitude.
  - **lr = 1e-4**: Measured by the "movement budget" (lr × number of update steps, an approximation of the cumulative displacement of the weights under Adam updates), even the most aggressive configuration, 40-clip full fine-tuning, has a budget of only about 2.3% of that of the pretraining stage, i.e., "adjustment" rather than "retraining". If the pretraining value of 1e-3 were reused, the 40-clip full fine-tuning budget would reach about 23%, enough to wash out the pretrained representation. The difference between Stage 2 (fine-tuning) and Stage 3 (no-pretraining control) would then be erased, the two stages' results would converge, and the control would lose its meaning.
  - **epochs = 30**: Derived backward from the step budget above. At this number of epochs the 5-clip level gets about 750 updates (enough for a small model to converge) and the 40-clip level gets about 5,820 updates (movement budget still below 2.5%, staying within the "adjustment" regime).

  **All four layer-adaptation strategies use the same set of hyperparameters and are not tuned separately.** If the optimal lr were searched for each strategy individually, RQ3 would actually compare the combined effect of "strategy + how carefully that strategy was tuned" rather than the properties of the strategy itself, and the conclusion would not be trustworthy.

  **No early stopping is used**, for three reasons: (1) the 5-clip level has too little data to carve out a usable validation subset, and holding out validation samples would break the nested-sampling contract of Section 4.3 (5/10/20/40 would be forced to become 4/8/16/32, and the "data-volume level" independent variable would no longer be clean); (2) early stopping would make the actual number of training steps vary across configurations, destroying comparability among the 272 configurations; (3) validation reconstruction MSE is not a valid proxy for AUC. An autoencoder's reconstruction loss on in-domain normal samples keeps decreasing while its ability to discriminate anomalies may already be degrading; the two are not monotonically related, so early stopping on validation MSE amounts to optimizing a quantity unrelated to the research objective. Early stopping on performance on the target-domain test round, meanwhile, is data leakage explicitly forbidden by LORO.

- No source-domain data replay

- **Implementation requirement**: all features must be extracted once and cached as .npy files; librosa must not be called again to extract features at every training/evaluation run. With 1,032 training-evaluation runs, re-extracting features each time would make redundant computation the main bottleneck of the actual run time.

**Starting-model and target-domain combinations:**

| Part | Experimental configurations | LORO folds | Actual training-evaluation runs |
| --- | --- | --- | --- |
| Small shift: 4 starting points × 3 targets × 4 data volumes × 4 layer strategies | 192 | 1 (LORO not applicable) | 192 |
| Large shift: 5 starting points × 1 target × 4 data volumes × 4 layer strategies | 80 | 10 | 800 |
| **Stage subtotal** | **272** |  | **992** |

**On the distinction between "configurations" and "training runs":** 272 is the number of **experimental configurations** (variable combinations), not the number of actual training runs. Because each large-shift configuration must run all 10 LORO folds, and each fold must restart fine-tuning from the pretrained model, the actual number of training-evaluation runs reaches 992. Adding Stage 3's 40 runs, the whole project involves about **1,032 training-evaluation runs**. This directly affects the time budget and output volume. It is recommended to persist only each configuration's evaluation results (CSV, under `results/`) and discard intermediate model weights after use; otherwise over a thousand .pth files would be produced.

**Metrics evaluated for each configuration:**

- Target domain: AUC + Best F1 (adaptation effect)
- Source domain: AUC + Best F1 (degree of forgetting, obtained by subtracting the Stage 0 source-domain baseline)
- The large-shift part is aggregated with both the pooled and per-fold schemes.

The direct comparison of **merged model vs. single-ID models** on the "adapt to the target-domain fan" pairs is used to answer the RQ3 question of whether pretraining diversity reduces the data required for adaptation.

### Stage 3: No-Pretraining Control

A randomly initialized model is trained directly on the self-recorded fan recordings only (without any MIMII pretraining). A randomly initialized model has no notion of a "learned encoder/decoder", so there is no layer-strategy variable; one model is trained for each of the 4 data-volume levels (5/10/20/40), for **4 experimental configurations in total**, all with full training.

This stage runs on the target domain and **must follow the LORO protocol**: 4 configurations × 10 folds = **40 training-evaluation runs**.

**Metric note:** The models in this stage have never seen MIMII data, so **there is no source domain; source-domain metrics are marked N/A**, and only target-domain AUC and Best F1 are reported.

**Note on expected results:** A randomly initialized model trained on 5 clips (about 1,500 320-dimensional feature vectors) is very likely to overfit until its reconstruction error approaches zero and its discriminative ability is close to random guessing. **This is not an implementation error but direct evidence that pretraining has value**; it should be recorded as is and not treated as a bug to be debugged repeatedly.

## 8. Expected Outputs

- **Domain-shift quantification figure**: comparison of the AUC and threshold-shift-factor gaps between the small and large shifts (addresses RQ1). Since the source and target evaluation sets share the 1:2 ratio, F1 can also serve as a secondary comparison, but AUC is primary.
- **Data-efficiency curves**: target-domain AUC as a function of fine-tuning data volume, grouped by layer strategy and shift magnitude (addresses RQ2).
- **Trade-off analysis figure**: how each layer strategy trades "target-domain gain" against "source-domain forgetting" (addresses RQ3).
- **Pretraining diversity effect**: the difference between the merged and single-ID models in large-shift adaptation (addresses RQ3).
- **Validation of pretraining value**: comparison of adaptation with and without pretraining.
- **LORO stability report**: per-fold standard deviations of the main configurations, supporting the reliability of the pooled results.

## 9. Limitations (stated up front)

- The target domain consists of only one consumer-grade fan and one microphone; the conclusions are a case study, not a general law.
- The four MIMII IDs all belong to the same industrial fan product line, so the "between-ID shift" is still much smaller than the "industrial-to-consumer" shift; the two are not comparisons of the same order of magnitude.
- The total of 180 recordings is limited; even with LORO validation, correlation among recordings from the same batch cannot be fully ruled out.
- Each LORO fold's test set has only 18 clips (6 normal + 12 anomalous), so per-fold metrics have high variance.
- **The pooled scheme does not apply per-fold normalization**: in Stages 2 and 3, the 10 LORO folds each produce an independent fine-tuned model, and the absolute scale of reconstruction error may differ between models. Pooling scores from different models and ranking them together may misrecord cross-fold scale differences as discrimination errors, systematically underestimating the pooled metrics. This study keeps the per-fold scheme as a check; if the two schemes differ significantly, this must be explained in the results analysis. Per-fold normalization (dividing by the median MSE of each fold's training-side normal samples) is listed as a future improvement.
- No experiment is independently repeated (a single run with seed = 42 only); if time permits, the best configurations may be repeated with multiple seeds to confirm stability.
- The four single-ID models do not have exactly equal numbers of training samples (851 to 873; see Section 4.1), and these are not forcibly equalized.
- **A fixed number of epochs gives different data-volume levels different numbers of gradient updates**: with epochs fixed at 30, the 40-clip level gets about 7.8 times as many updates as the 5-clip level (about 5,820 vs. 750). The conclusion "40 clips beats 5 clips" on the data-efficiency curve therefore mixes in a contribution from "training longer" rather than purely "more data". Fixing the number of update steps instead would remove this confound but at a higher cost: fixing at the 5-clip level's 750 steps would let the 40-clip level train for only about 3.1 epochs, severely undertraining it, artificially flattening the data-efficiency curve and in turn undermining the RQ2 conclusions. Weighing the two drawbacks, fixed epochs is chosen.
- **The batch size differs between pretraining and fine-tuning** (512 vs. 64), so the optimization dynamics of the two stages (gradient-noise level, effective learning rate) are not fully comparable, and performance differences in the fine-tuning stage may include an effect of the batch-size switch itself.
- The metrics used (AUC and Best F1) are threshold-independent and reflect an upper bound on the model's ranking ability. In real deployment the threshold must be estimated from a limited number of normal samples, so actual performance will be lower than the numbers reported here.
- **Best F1 has a trivial floor of 0.800**: the source and target evaluation sets share the 1:2 ratio, so their floors are identical (both 0.800) and F1 can be compared across shifts. Note, however, that this floor is itself high: no model's F1 will fall below 0.800. This deserves particular attention for the Stage 3 no-pretraining control, whose F1 will appear as about 0.80; this is the floor value of a zero-information model, not actual discriminative ability, and it should always be read alongside AUC (floor 0.5).

## 10. Future Work (optional, for future extension)

- **Threshold calibration**: this study's metrics are threshold-independent and do not address how to estimate a deployment threshold from a small number of normal samples. This direction could be a standalone study of the stability of different calibration rules (quantile-based, median-multiple, etc.) under limited data.
- **Per-fold normalization**: before pooling, normalize by the median MSE of each fold's training-side normal samples to remove cross-fold scale drift.
- A systematic sweep of learning rate as an independent variable.
- Source-domain data replay to mitigate forgetting.
- Validation on more target hardware (currently only one consumer-grade fan).
- Domain-adversarial training (DANN) as a comparison with a more complex adaptation method.
- Multi-seed repeated experiments to quantify the range of variability in the results.
