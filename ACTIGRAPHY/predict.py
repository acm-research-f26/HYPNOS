"""Inference on new Apple Watch accelerometer files (no PSG labels necessary).
Example: python ACTIGRAPHY/predict.py path/to/123_acceleration.txt
Note: expects seconds relative to PSG start or a chosen recording origin;
output times are relative to whatever timestamp units were supplied.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
try:
    from ACTIGRAPHY.model import ActigraphyCNN
except ModuleNotFoundError:
    from model import ActigraphyCNN

HERE = Path(__file__).resolve().parent


def predict_file(path, output):
    data = pd.read_csv(path, sep=r'\s+', header=None, names=['t', 'x', 'y', 'z'], dtype='float32')
    data = data.replace([np.inf, -np.inf], np.nan).dropna().sort_values('t').drop_duplicates('t')
    t = data['t'].to_numpy(dtype=np.float64)
    xyz = data[['x', 'y', 'z']].to_numpy(dtype=np.float32)
    if len(t) == 0:
        raise ValueError('No valid readings')
    params = np.load(HERE / 'models' / 'normalization.npz')
    mean, std = params['mean'], params['std']
    model = ActigraphyCNN()
    model.load_state_dict(torch.load(HERE / 'models' / 'actigraphy_cnn.pth',
                                     map_location='cpu', weights_only=True))
    model.eval()
    # Use 30-second grid from first reading; timestamps must be on same scale.
    starts = np.arange(t[0], t[-1]-30+1e-6, 30.)
    rows = []
    with torch.no_grad():
        for start in starts:
            grid = start + np.arange(300) / 10.
            left = max(0, np.searchsorted(t, start)-1)
            right = min(len(t), np.searchsorted(t, start+30)+1)
            local_t = t[left:right]
            if len(local_t) < 180 or local_t[0] > grid[0] or local_t[-1] < grid[-1]:
                continue
            if np.diff(local_t).max(initial=0) > 3.:
                continue
            window = np.stack([np.interp(grid, local_t, xyz[left:right, i])
                               for i in range(3)], axis=0).astype(np.float32)
            x = torch.from_numpy(((window[None] - mean)/std).astype(np.float32))
            score = float(torch.sigmoid(model(x)).item())
            rows.append({'subject_id': path.name.split('_')[0],
                         'window_start_sec': start, 'window_end_sec': start+30,
                         'p_sleep': score, 'pred_sleep': int(score >= .5)})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['subject_id', 'window_start_sec', 'window_end_sec',
                                'p_sleep', 'pred_sleep']).to_csv(output, index=False)
    print(f'Saved {len(rows)} windows to {output}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('acceleration_file', type=Path)
    ap.add_argument('--out', type=Path, default=HERE / 'outputs' / 'new_predictions.csv')
    args = ap.parse_args()
    predict_file(args.acceleration_file, args.out)
