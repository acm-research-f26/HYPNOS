# HYPNOS wrist actigraphy: ready-to-run 1D CNN pipeline

Place `model.py`, `preprocess.py`, `run_pipeline.py`, and `predict.py` under the existing `HYPNOS/ACTIGRAPHY/` directory. This bundle replaces the starter `preprocess.py` and `model.py`; it **does not need to replace your existing `train.py`**, since `run_pipeline.py` has a self-contained training loop.

**From repository root (`C:\Users\urmip\HYPNOS`) with virtual environment active:**

1. Extract the full PhysioNet SleepAccel dataset under `ACTIGRAPHY/data/raw/`. Any nested layout works as long as `*_acceleration.txt` and `*_labeled_sleep.txt` are present.
2. `python ACTIGRAPHY/preprocess.py`
3. `python ACTIGRAPHY/run_pipeline.py`
4. Results are written to `ACTIGRAPHY/outputs/metrics.json` and `ACTIGRAPHY/outputs/sleep_predictions_test.csv`; weights are in `ACTIGRAPHY/models/actigraphy_cnn.pth`.
5. On a **new unlabeled file** with same 4-column motion format: `python ACTIGRAPHY/predict.py "path/to/123_acceleration.txt"`.

Preprocessing: Uses 30-second PSG labels to construct 30-second windows. Resamples XYZ at 10 Hz by linear interpolation; drops missing/gapped windows. Subject-level 70/15/15 approximate split. Normalization is fitted **on training subjects only**. Wake=0, sleep=1. `p_sleep` is uncalibrated, and threshold is 0.5 by default.

**Limits**: SleepAccel excludes patients with sleep apnea. This is a research prototype, not a diagnosis. More tuning and validation on OSA-relevant data are needed. PSG labels and accelerometer intervals are assumed aligned per published time convention. The model predicts from a single window, with no long-duration temporal context. Performance may be weaker for still-awake periods. Avoid reporting high accuracy without inspecting held-out data.

**Git:** Do not commit raw/processed data or `.venv`. Your current `.gitignore` excludes the checkpoint file extensions, but add `ACTIGRAPHY/models/*.npz` if you want to ignore training normalization statistics as well. Commit only source files and the README.
