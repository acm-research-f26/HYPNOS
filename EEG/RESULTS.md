# Single-channel EEG apnea/hypopnea detection — results

Task: per-30 s-window apnea/hypopnea detection from one central EEG channel. Threshold = max-MCC on validation, applied unchanged to every test set. ρ = Spearman correlation of per-patient predicted vs. true burden.

## Winner: **F — context model (prev + current + next window, 90 s)**
Shared 1D-ResNet encoder on each of the 3 windows, concatenated into an MLP. Trained on all 796 train patients, 8 epochs, best val AUROC 0.800.

| Set | patients | prevalence | AUROC | MCC | Sens | Spec | F1 | burden ρ |
|---|---|---|---|---|---|---|---|---|
| CinC internal test | 99 | 12.8% | **0.814** | 0.365 | 0.52 | 0.89 | 0.45 | 0.50 |
| UCDDB | 25 | 15.3% | 0.771 | 0.275 | 0.30 | 0.93 | 0.36 | 0.56 |
| MIT-BIH slpdb | 18 | 32.8% | 0.691 | 0.214 | 0.37 | 0.82 | 0.43 | 0.51 |
| DREAMS | 12 | 34.9% | 0.777 | 0.340 | 0.42 | 0.88 | 0.51 | 0.25 |

Baseline (results_v3): AUROC 0.65 on 1 UCDDB patient. Now 0.77 on 25 UCDDB patients.

## Screen: every architecture (200 train patients, 3 epochs) × every test set — AUROC, all windows

| Model | val | CinC test | UCDDB | MIT-BIH | DREAMS |
|---|---|---|---|---|---|
| A  Barnes CNN | 0.700 | 0.704 | 0.703 | 0.658 | 0.737 |
| B  1D ResNet | 0.712 | 0.715 | 0.701 | 0.653 | 0.728 |
| C  CNN + BiGRU | 0.725 | 0.727 | 0.709 | 0.680 | 0.727 |
| D  STFT + 2D CNN | 0.694 | 0.691 | 0.703 | 0.659 | 0.737 |
| E  band power + logistic reg. | 0.668 | 0.669 | 0.666 | 0.646 | 0.681 |
| E  band power + grad. boosting | 0.685 | 0.683 | 0.694 | 0.651 | 0.688 |
| **F  3-window context** | **0.749** | **0.760** | **0.728** | 0.673 | **0.769** |

## Full training (all patients) — top 2 from the screen

| Model | val | CinC test | UCDDB | MIT-BIH | DREAMS | ρ test | ρ UCDDB | ρ MIT-BIH | ρ DREAMS |
|---|---|---|---|---|---|---|---|---|---|
| C  CNN + BiGRU | 0.777 | 0.785 | 0.776 | 0.679 | 0.777 | 0.41 | 0.69 | 0.57 | 0.30 |
| **F  context** | **0.800** | **0.814** | 0.771 | **0.691** | 0.777 | **0.50** | 0.56 | 0.51 | 0.25 |

Sleep-only (wake windows removed) AUROC is lower: F 0.790 / 0.746 / 0.636 / 0.756, C 0.756 / 0.750 / 0.620 / 0.751 (test / UCDDB / MIT-BIH / DREAMS). Part of the all-window score comes from separating wake from sleep. Full tables: `results_final/metrics.json` (`sleep_only` per set).

## Why F won
- Ranked first or tied-first on val and on 3 of 4 test sets in the screen, and on internal test, MIT-BIH and val after full training. Events cross window edges, so ±30 s of context helps.
- On UCDDB and DREAMS, C ties F on AUROC and is better at UCDDB burden ρ (0.69 vs 0.56). On those two sets the margin is within noise given 25 and 12 patients.
- F costs ~6× the training time of C (≈5.5 vs ≈1 min/epoch).

## Follow-up: improving F, and an ensemble
Screen on 200 patients / 3 epochs, val AUROC (two seeds where noted; seed-to-seed spread ≈ 0.006):

| Variant | val AUROC |
|---|---|
| F (control) | 0.749 / 0.755 |
| F + augmentation (polarity, gain, noise) | 0.740 |
| **F + sleep-stage side loss (weight 0.2)** | **0.760 / 0.764** |
| F + sleep-stage side loss (weight 1.0) | 0.752 |
| ±2 windows of context (150 s) | 0.751 |
| ±2 windows + BiGRU over window embeddings | not finished (5× slower per epoch on MPS) |

The side loss (predict the centre window's sleep stage from its embedding) helped on both seeds, so it was trained on all 796 patients for up to 14 epochs (`full_F3_aux`, early stop after epoch 12, best val AUROC 0.811 vs F's 0.800).

**It did not transfer.** On held-out data it matched F on AUROC (test 0.817, UCDDB 0.770, MIT-BIH 0.684, DREAMS 0.778) and its val-chosen threshold was stricter, so external sensitivity and burden ρ fell (MIT-BIH ρ 0.51 → 0.22, DREAMS 0.25 → 0.05). F stays the best single model.

**Ensemble F + `full_F3_aux` + C** (mean of each model's scores standardised with its own val mean/std) is better than any one model on every set. Mean external AUROC 0.768 vs 0.746 for F alone. Choosing the ensemble members was done after seeing test scores, and the external sets have 12–25 patients, so treat differences under ≈ 0.01 as noise.

Threshold = max MCC on val, applied unchanged:

| Set | patients | prevalence | AUROC | MCC | Sens | Spec | F1 | burden ρ | sleep-only AUROC |
|---|---|---|---|---|---|---|---|---|---|
| CinC internal test | 99 | 12.8% | 0.827 | 0.385 | 0.50 | 0.91 | 0.47 | 0.51 | 0.803 |
| UCDDB | 25 | 15.3% | 0.803 | 0.324 | 0.34 | 0.94 | 0.40 | 0.62 | 0.775 |
| MIT-BIH slpdb | 18 | 32.8% | 0.697 | 0.203 | 0.33 | 0.84 | 0.40 | 0.48 | 0.633 |
| DREAMS | 12 | 34.9% | 0.804 | 0.328 | 0.35 | 0.91 | 0.46 | 0.27 | 0.779 |

Same ensemble, threshold = max (sensitivity + specificity) on val. This does not depend on prevalence, so it transfers better to the external sets, which have 15–35% prevalence against 12% in CinC. It trades specificity for sensitivity, which suits a screening use:

| Set | AUROC | MCC | Sens | Spec | F1 | burden ρ |
|---|---|---|---|---|---|---|
| CinC internal test | 0.827 | 0.350 | 0.76 | 0.73 | 0.42 | 0.35 |
| UCDDB | 0.803 | 0.343 | 0.70 | 0.74 | 0.45 | 0.73 |
| MIT-BIH slpdb | 0.697 | 0.296 | 0.64 | 0.67 | 0.55 | 0.41 |
| DREAMS | 0.804 | 0.446 | 0.73 | 0.73 | 0.66 | 0.34 |

Reproduce: `python apnea.py report F_full full_F3_aux C_full` (add `--plots file.png` for ROC, confusion and burden plots).

## Matching the device: band-pass ablations
Target hardware records roughly 5–50 Hz with a Butterworth band-pass. The CinC cache was low-passed at 42 Hz and the raw CinC (200 Hz) is not on disk, so the closest available band is 5–42 Hz. `--hp HZ` adds a 4th-order Butterworth high-pass at load time (then re-z-scores each window); `--causal` makes it forward-only. All models are F + sleep-stage side loss, full data, one run each, threshold = max MCC on val. Differences under ≈ 0.01 AUROC are noise (single runs, external sets of 12–25 patients).

**1. High-pass at 5 Hz (zero-phase).** Removing everything below 5 Hz costs nothing internally and helped on other hospitals (hypothesis, untested: the lowest frequencies carry recording-specific drift).

| Set | AUROC 0.5–42 Hz → 5–42 Hz | MCC | burden ρ |
|---|---|---|---|
| val (CinC) | 0.811 → 0.809 | – | – |
| CinC test | 0.817 → 0.820 | 0.37 → 0.38 | 0.54 → 0.52 |
| UCDDB | 0.770 → 0.804 | 0.27 → 0.39 | 0.56 → 0.70 |
| MIT-BIH | 0.684 → 0.711 | 0.18 → 0.27 | 0.22 → 0.64 |
| DREAMS | 0.778 → 0.781 | 0.27 → 0.32 | 0.05 → 0.14 |

**2. Causal (forward-only) vs zero-phase high-pass.** Matched training and scoring filter gives the same accuracy; a mismatch is a large loss in both directions. Val AUROC of the causal model: 0.812.

| Trained on → scored on | CinC test | UCDDB | MIT-BIH | DREAMS |
|---|---|---|---|---|
| zero-phase → zero-phase | 0.820 | 0.804 | 0.711 | 0.781 |
| causal → causal | 0.821 | 0.805 | 0.710 | 0.766 |
| zero-phase → causal | 0.654 | 0.627 | 0.618 | 0.634 |
| causal → zero-phase | 0.673 | 0.662 | 0.619 | 0.571 |

A causal filter is fine, but the training pipeline must use the same filter as the device (same order, same cutoffs, same causality). Zero-phase training followed by causal deployment loses about 0.17 AUROC. The causal windows here are filtered per window from zero state, so the first ≈ 0.5 s of each window carries a start-up transient; the device filter runs continuously.

**3. Upper edge 42 → 50 Hz at test time.** The external sets were rebuilt from raw with a 50 Hz low-pass (`prep_external.py --lowpass 50`) and scored by the models trained on 5–42 Hz CinC (which never saw 42–50 Hz content).

| Model | UCDDB | MIT-BIH | DREAMS |
|---|---|---|---|
| zero-phase, 42 Hz → 50 Hz | 0.804 → 0.765 | 0.711 → 0.688 | 0.781 → 0.780 |
| causal, 42 Hz → 50 Hz | 0.805 → 0.772 | 0.710 → 0.679 | 0.766 → 0.765 |

Passing 42–50 Hz to a model trained without it costs 0.03–0.04 AUROC on UCDDB and 0.02–0.03 on MIT-BIH. UCDDB was recorded with 50 Hz mains, which a 50 Hz low-pass lets through; whether mains explains the drop was not tested. To use a 5–50 Hz device band, train on data filtered to the same band (needs the raw CinC again) and add a mains notch.

## Multi-channel: a channel-flexible model
The hardware has several channels, so the model should accept any number. `MC` (`--mc`) runs the same shared ResNet encoder on every channel, averages the channel embeddings (masked, so any subset works) and feeds the usual prev/current/next head. During training one channel is hidden at random with probability 0.3 (channel dropout). Channels here: C4-M1 and C3-M2 from CinC (`build_cache_c3.py`). UCDDB, MIT-BIH and DREAMS were prepared with one channel only, so their rows are single-channel for both models.

Screen (200 patients, 3 epochs, no high-pass), AUROC: one model scored three ways: both channels val 0.776 / test 0.782; C4 only 0.765 / 0.771; C3 only 0.767 / 0.772 (single-channel F with the side loss: val 0.760–0.764).

Full comparison, 5 Hz zero-phase high-pass, side loss, **400 training patients** (the 796-patient two-channel run needed 14 GB and stalled on swap, so both models use 400). Single run each; the 400-patient models are not comparable with the 796-patient rows above. AUROC, threshold = max MCC on val:

| Model | val | CinC test | UCDDB | MIT-BIH | DREAMS |
|---|---|---|---|---|---|
| single channel (C4) | 0.791 | 0.802 | 0.778 | 0.689 | 0.784 |
| MC, C4 only | 0.813 | 0.806 | 0.791 | 0.693 | 0.807 |
| MC, C3 only | – | 0.809 | – | – | – |
| MC, C3 + C4 | 0.813 | **0.821** | 0.791 | 0.693 | 0.807 |

- The channel-flexible model beats the single-channel model with the same data even when it is given only C4 (test 0.806 vs 0.802, external +0.003 to +0.023). Channel dropout seems to be a useful regulariser; this was not isolated.
- A second channel adds about 0.015 test AUROC (0.806 → 0.821, MCC 0.35 → 0.38). The test set has 99 patients and one run per model, so this is suggestive, not established.
- Either channel alone works, so a device that loses a channel degrades gracefully instead of failing.

## Caveats
- Absolute performance is modest: sensitivity 0.30–0.52 at the val-chosen threshold. External MCC is 0.21–0.34.
- The val-chosen threshold (raw logit 0.054) transfers poorly in prevalence: external sets have 15–35% prevalence vs 12% in CinC, so sensitivity drops there.
- MIT-BIH labels mean "event present in epoch", not ≥10 s overlap, so prevalence is higher and labels are noisier. By channel (F): C3-O1 AUROC 0.70 (11 rec), C4-A1 0.69 (5 rec), O2-A1 0.54 (2 rec; occipital, not central). Two MIT-BIH records have zero positives. Burden ρ on 2–5 records is not meaningful.
- Burden ρ with 12–25 patients has wide confidence intervals; DREAMS ρ 0.25 is not significant.
- Screen models used 200 patients/3 epochs, so the screen ranking of A/B/D/E may change with full training. Only C and F were fully trained.
- UCDDB stage coding (0=W, 1=REM, 2–5=S1–S4) follows the PhysioNet docs; only the wake code matters for sleep-only metrics. DREAMS stage = majority of six 5 s epochs.

## Cleaning (CinC2018)
919,246 → 889,988 windows. Dropped: 27,046 unscored (stage ""), 1,965 flat (z-scored std < 0.5), 247 stuck/clipped (>50 % identical consecutive samples), 0 non-finite. Patient split 80/10/10 seed 0: 796 / 99 / 99 patients. External sets also drop unscored/flat windows.

## Files
`prep_external.py` (UCDDB / MIT-BIH / DREAMS to windows; `--lowpass HZ --out DIR` for other bands), `build_cache.py` (clean + float16 memmap + split), `build_cache_c3.py` (second channel), `apnea.py` (data, models, training, scoring, ensemble reports), `preprocess.py` (helpers). Plots for model F are in `plots/`. Data, checkpoints and raw window scores are not included.
Note: UCDDB was preprocessed with the helper functions from preprocess.py rather than `process_subject` (which prints a lot and doesn't save stage).
