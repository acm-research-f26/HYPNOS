"""After extracting SleepAccel, run from HYPNOS root:
python ACTIGRAPHY/preprocess.py
python ACTIGRAPHY/run_pipeline.py
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

try:
    from ACTIGRAPHY.model import ActigraphyCNN
except ModuleNotFoundError:
    from model import ActigraphyCNN

HERE = Path(__file__).resolve().parent


def load_split(name):
    f = HERE / 'data' / 'processed' / f'{name}.npz'
    if not f.exists():
        raise FileNotFoundError(f'{f} missing; run preprocess.py first')
    with np.load(f) as data:
        return {k: data[k] for k in ('x', 'y', 'id', 't')}


def main(epochs, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train, val, test = (load_split(s) for s in ('train', 'val', 'test'))
    train_subjects, val_subjects, test_subjects = (set(x['id']) for x in (train, val, test))
    assert train_subjects.isdisjoint(val_subjects | test_subjects)
    assert val_subjects.isdisjoint(test_subjects)
    mean = train['x'].mean(axis=(0, 2), keepdims=True)
    std = train['x'].std(axis=(0, 2), keepdims=True).clip(min=1e-3)
    norm = lambda d: ((d['x'] - mean) / std).astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def loader(d, shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(norm(d)),
                                        torch.from_numpy(d['y'].astype(np.float32)).unsqueeze(1)),
                          batch_size=batch_size, shuffle=shuffle)
    train_loader, val_loader, test_loader = loader(train, True), loader(val), loader(test)
    model = ActigraphyCNN().to(device)
    positives = float(train['y'].sum())
    negatives = float(len(train['y']) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError('Training split has only one class; cannot train sleep/wake model.')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = HERE / 'models' / 'actigraphy_cnn.pth'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                loss = criterion(model(x.to(device)), y.to(device))
                val_loss += loss.item() * len(y)
        train_loss /= len(train['y'])
        val_loss /= len(val['y'])
        print(f'Epoch {epoch+1}/{epochs}: train loss={train_loss:.4f}, val loss={val_loss:.4f}', flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    probabilities = []
    with torch.no_grad():
        for x, _ in test_loader:
            probabilities.append(torch.sigmoid(model(x.to(device))).cpu().numpy().ravel())
    p = np.concatenate(probabilities)
    y = test['y']
    pred = (p >= .5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    wake_recall = float(cm[0, 0] / max(1, cm[0].sum()))
    sleep_recall = float(cm[1, 1] / max(1, cm[1].sum()))
    metrics = {'accuracy': float(accuracy_score(y, pred)),
               'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
               'f1_sleep': float(f1_score(y, pred, zero_division=0)),
               'wake_specificity': wake_recall,
               'sleep_sensitivity': sleep_recall,
               'auroc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
               'test_subjects': sorted(str(v) for v in test_subjects),
               'confusion_matrix_rows_awake_sleep': cm.tolist(),
               'threshold': .5, 'sleep_label': 1}
    output = HERE / 'outputs'
    output.mkdir(exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({'subject_id': test['id'], 'window_start_sec': test['t'],
                  'window_end_sec': test['t']+30, 'p_sleep': p,
                  'true_sleep': y, 'pred_sleep': pred}).to_csv(output / 'sleep_predictions_test.csv', index=False)
    np.savez(HERE / 'models' / 'normalization.npz', mean=mean, std=std)
    print('\nHeld-out subject results (not clinical validation):')
    print(json.dumps(metrics, indent=2))
    print('Saved checkpoint, train-only scaling stats, test predictions and metrics.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    options = ap.parse_args()
    main(options.epochs, options.batch_size)
