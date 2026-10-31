"""Score ensembles and threshold rules from saved scores (CPU only, no model runs).
  .venv/bin/python ens_eval.py
Ensemble = mean of each model's scores after standardising with that model's VAL mean/std.
Threshold rules, both chosen on val only:  mcc = max MCC (what the report uses);  bal = max balanced accuracy
(sensitivity + specificity), which does not depend on prevalence, so it should move less between hospitals."""
import numpy as np
from sklearn.metrics import roc_curve

import common as C

SETS = ["test", "ucddb", "mitbih", "dreams"]
MODELS = {"F": "F_full", "N": "full_F3_aux", "C": "C_full"}
D = {k: np.load(f"results_final/scores/{v}.npz") for k, v in MODELS.items()}


def scores(combo):
    out = {}
    for s in ["val"] + SETS:
        zs = []
        for m in combo:
            mu, sd = D[m]["val__s"].mean(), D[m]["val__s"].std()
            zs.append((D[m][f"{s}__s"] - mu) / sd)
        out[s] = np.mean(zs, 0)
    return out


def bal_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


print(f"{'models':8s} {'rule':4s} " + " ".join(f"{k:>6s}auc {k:>5s}sen {k:>5s}rho" for k in SETS) + "   mean-auc  mean-rho")
for combo in ("F", "N", "FN", "FNC"):
    s = scores(combo)
    y = {k: D["F"][f"{k}__y"] for k in ["val"] + SETS}
    rec = {k: D["F"][f"{k}__rec"] for k in SETS}
    for rule in ("mcc", "bal"):
        thr = C.pick_threshold(y["val"], s["val"]) if rule == "mcc" else bal_threshold(y["val"], s["val"])
        ms = {k: C.metrics(y[k], s[k], thr, rec[k]) for k in SETS}
        line = " ".join(f"{ms[k]['auroc']:9.3f} {ms[k]['sensitivity']:9.2f} {ms[k]['burden_spearman']:9.2f}" for k in SETS)
        ext = SETS[1:]
        print(f"{combo:8s} {rule:4s} {line}   {np.mean([ms[k]['auroc'] for k in ext]):.3f}     "
              f"{np.mean([ms[k]['burden_spearman'] for k in ext]):.3f}")
