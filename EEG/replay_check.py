"""End-to-end check of device_pipeline.py: MIT-BIH slpdb is recorded at 250 Hz like the Cerelog board, so replay its raw
signal through the device path (causal filter, notch, decimate, rolling scorer) and compare AUROC with the scores the
same model got from the training-style preparation (saved by `apnea.py score`).

    python replay_check.py MODEL [--scores SAVED_SCORES_NAME] [--mains 50]
"""
import argparse
from pathlib import Path

import numpy as np
import wfdb
from sklearn.metrics import roc_auc_score

import apnea
import device_pipeline as D

RAW, EXT = Path("data/raw/mitbih_slpdb"), Path("data/processed/external")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--scores", help="saved score file of the same model (default: same name)")
    p.add_argument("--mains", type=float, default=60.0)
    a = p.parse_args()
    model = apnea.load(a.model)
    saved = np.load(apnea.find(a.scores or a.model, "npz"))
    ys, dev, ref = [], [], []
    for rec in (RAW / "RECORDS").read_text().split():
        d = np.load(EXT / f"mitbih_{rec}.npz")
        h = wfdb.rdheader(str(RAW / rec))
        ch = next(i for i, n in enumerate(h.sig_name) if n.startswith("EEG"))
        assert h.fs == D.FS_IN
        x = np.nan_to_num(wfdb.rdrecord(str(RAW / rec), channels=[ch]).p_signal) * (1000 if h.units[ch] == "mV" else 1)
        pre, scorer, res = D.Preprocessor(1, mains=a.mains, warmup_s=0), D.Scorer(model), []
        for i in range(0, len(x), 1000):  # arbitrary chunking, like a network stream
            for w in pre.push(x[i:i + 1000]):
                res += scorer.push(w)
        res += scorer.flush()
        logit = np.full(len(d["y"]), np.nan)
        for r in res:
            if r["index"] < len(logit):
                logit[r["index"]] = r["logit"]
        ok = (d["stage"] != "") & np.isfinite(d["X"]).all(1) & (d["X"].std(1) > 0.5)  # the rows apnea.external() keeps
        sel = np.flatnonzero(ok)
        k = np.isfinite(logit[sel])
        r_ = saved["mitbih__rec"] == f"mitbih_{rec}"
        ys.append(d["y"][sel][k]); dev.append(logit[sel][k]); ref.append(saved["mitbih__s"][r_][k])
        print(f"{rec}: {k.sum()} of {len(sel)} windows scored on the device path", flush=True)
    y, dev, ref = map(np.concatenate, (ys, dev, ref))
    print(f"\nMIT-BIH slpdb, {len(y)} windows, model {a.model}")
    print(f"AUROC, training-style preparation: {roc_auc_score(y, ref):.3f}")
    print(f"AUROC, device pipeline (causal):   {roc_auc_score(y, dev):.3f}")
    print(f"score correlation between the two: {np.corrcoef(dev, ref)[0, 1]:.3f}")
