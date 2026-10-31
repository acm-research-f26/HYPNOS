"""Score val + internal test + the three external sets with trained models and save raw scores.
  .venv/bin/python evaluate_new.py A_screen B_screen ... E_screen      (name = <arch>_<tag>)
Output: results_final/scores/<name>.npz  (per set: y, s, rec, stage[, channel]); E expands to E_lr_<tag> / E_gbm_<tag>."""
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

import common as C
from models_new import CONTEXT, make_model
from train_new import DEV, OUT, predict

SC = Path("results_final/scores")


def sets():
    meta, X_all = C.load_meta(), C.open_all()
    d = {"val": C.gather(X_all, meta, C.split_rows(meta, 1)), "test": C.gather(X_all, meta, C.split_rows(meta, 2))}
    d.update(C.load_external())
    return d


def save(name, scores, data):
    arrs = {}
    for k, ws in data.items():
        arrs.update({f"{k}__y": ws.y, f"{k}__s": scores[k], f"{k}__rec": ws.rec, f"{k}__stage": ws.stage})
        if hasattr(ws, "channel"):
            arrs[f"{k}__channel"] = ws.channel
    np.savez(SC / f"{name}.npz", **arrs)
    print("saved", name, flush=True)


if __name__ == "__main__":
    SC.mkdir(parents=True, exist_ok=True)
    data = sets()
    feats = {}
    for name in sys.argv[1:]:
        arch, tag = name.split("_", 1)
        if arch == "E":
            if not feats:
                feats = {k: C.features(ws.X) for k, ws in data.items()}
            m = joblib.load(OUT / f"E_{tag}.joblib")
            save(f"E_lr_{tag}", {k: m["lr"].decision_function(m["scaler"].transform(f)) for k, f in feats.items()}, data)
            save(f"E_gbm_{tag}", {k: m["gbm"].predict_proba(f)[:, 1] for k, f in feats.items()}, data)
        else:
            model = make_model(arch).to(DEV)
            model.load_state_dict(torch.load(OUT / f"{name}.pt", map_location=DEV))
            save(name, {k: predict(model, ws, arch in CONTEXT) for k, ws in data.items()}, data)
