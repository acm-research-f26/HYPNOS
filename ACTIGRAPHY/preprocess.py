"""Convert PhysioNet SleepAccel motion and PSG labels into patient-wise NPZ splits.

Run at repo root: python ACTIGRAPHY/preprocess.py

Uses 10 Hz, 30-s windows (300 acceleration readings x 3 axes). The CSV-like
text files contain whitespace-delimited columns and times in seconds from PSG
start; PSG sleep stages are scored every 30 seconds. Unknown stages are skipped.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
RATE = 10
WINDOW_SECONDS = 30
N_SAMPLES = RATE * WINDOW_SECONDS
VALID_STAGES = {0, 1, 2, 3, 5}


def extract_subject(motion_path: Path, labels_path: Path):
    # 4 columns in motion: seconds-from-PSG-start, x, y, z (g).
    # 2 columns in labels: seconds-from-PSG-start, PSG stage.
    motion = pd.read_csv(motion_path, sep=r'\s+', header=None,
                         names=['t', 'x', 'y', 'z'], dtype='float32')
    labels = pd.read_csv(labels_path, sep=r'\s+', header=None,
                         names=['t', 'stage'], dtype='float32')
    motion = motion.replace([np.inf, -np.inf], np.nan).dropna()
    labels = labels.replace([np.inf, -np.inf], np.nan).dropna()
    motion = motion.sort_values('t').drop_duplicates('t')
    labels = labels.sort_values('t').drop_duplicates('t')
    ts = motion['t'].to_numpy(dtype=np.float64)
    vals = motion[['x', 'y', 'z']].to_numpy(dtype=np.float32)
    label_times = labels['t'].to_numpy(dtype=np.float64)
    stages = labels['stage'].to_numpy()
    result_x, result_y, result_t = [], [], []
    grid_offset = np.arange(N_SAMPLES, dtype=np.float64) / RATE
    for start, stage in zip(label_times, stages):
        if stage not in VALID_STAGES:
            continue
        # Only windows fully contained in usable accelerometer recordings.
        grid = start + grid_offset
        left = np.searchsorted(ts, start, side='left')
        right = np.searchsorted(ts, start + WINDOW_SECONDS, side='left')
        if left >= len(ts) or right - left < N_SAMPLES * 0.6:
            continue
        # Include neighboring points so interpolation reaches both boundaries.
        lo = max(0, left - 1)
        hi = min(len(ts), right + 1)
        local_t = ts[lo:hi]
        if local_t[0] > grid[0] or local_t[-1] < grid[-1]:
            continue
        # Reject recording gaps >3 seconds; never invent long motion segments.
        if np.diff(local_t).max(initial=0) > 3.0:
            continue
        window = np.stack([np.interp(grid, local_t, vals[lo:hi, c])
                           for c in range(3)], axis=0).astype(np.float32)
        if not np.isfinite(window).all():
            continue
        result_x.append(window)
        result_y.append(0 if int(stage) == 0 else 1)
        result_t.append(float(start))
    if not result_x:
        return (np.empty((0, 3, N_SAMPLES), np.float32),
                np.empty((0,), np.int64), np.empty((0,), np.float64))
    return np.stack(result_x), np.array(result_y, np.int64), np.array(result_t)


def prepare(raw_dir: Path, processed_dir: Path):
    motion_files = sorted(raw_dir.rglob('*_acceleration.txt'))
    labels_files = {p.name.split('_')[0]: p for p in raw_dir.rglob('*_labeled_sleep.txt')}
    matched = [(m, labels_files[m.name.split('_')[0]]) for m in motion_files
               if m.name.split('_')[0] in labels_files]
    if len(matched) < 5:
        raise RuntimeError(f'Found only {len(matched)} matched subjects in {raw_dir}. '
                           'Need at least five participants for patient-independent splits.')
    subjects = [m.name.split('_')[0] for m, _ in matched]
    train_ids, other_ids = train_test_split(subjects, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(other_ids, test_size=0.50, random_state=42)
    assignment = {sid: group for group, ids in
                  [('train', train_ids), ('val', val_ids), ('test', test_ids)] for sid in ids}
    groups = {name: {'x': [], 'y': [], 'id': [], 't': []} for name in ('train', 'val', 'test')}
    for motion, label in matched:
        sid = motion.name.split('_')[0]
        x, y, t = extract_subject(motion, label)
        print(f'{sid}: {len(y)} usable labeled windows ({assignment[sid]})', flush=True)
        if not len(y):
            continue
        g = groups[assignment[sid]]
        g['x'].append(x)
        g['y'].append(y)
        g['id'].append(np.repeat(sid, len(y)))
        g['t'].append(t)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, g in groups.items():
        if not g['x']:
            raise RuntimeError(f'No usable recordings for {name} split; inspect data files.')
        payload = {key: np.concatenate(items) for key, items in g.items()}
        # Normalize by training-only per-axis statistics, computed later in pipeline.
        np.savez(processed_dir / f'{name}.npz', **payload)
        print(f'Wrote {name}.npz: {len(payload["y"])} windows, '
              f'{len(set(payload["id"]))} subjects')
    print('Preprocessing complete. PSG stage 0=awake, 1/2/3/5=asleep.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=HERE / 'data' / 'raw')
    p.add_argument('--processed', type=Path, default=HERE / 'data' / 'processed')
    args = p.parse_args()
    prepare(args.raw, args.processed)
