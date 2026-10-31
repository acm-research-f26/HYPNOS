"""Turn saved scores into metrics.json, tables, ROC / confusion / burden plots.
  .venv/bin/python report_new.py <tag> [winner_name]
Threshold = max-MCC on val, applied unchanged everywhere. Each set is reported on all windows and on sleep-only
(stage != W) windows, plus per-patient burden Spearman."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

import common as C

SC, OUT = Path("results_final/scores"), Path("results_final")
SETS = ["test", "ucddb", "mitbih", "dreams"]


def evaluate(path):
    d = np.load(path)
    thr = C.pick_threshold(d["val__y"], d["val__s"])
    res = {"threshold": thr, "val_auroc": C.metrics(d["val__y"], d["val__s"], thr)["auroc"]}
    for k in SETS:
        y, s, rec, st = d[f"{k}__y"], d[f"{k}__s"], d[f"{k}__rec"], d[f"{k}__stage"]
        res[k] = {"all": C.metrics(y, s, thr, rec)}
        m = st != "W"
        res[k]["sleep_only"] = C.metrics(y[m], s[m], thr, rec[m])
        if k == "mitbih":
            ch = d["mitbih__channel"]
            res[k]["by_channel"] = {c: C.metrics(y[ch == c], s[ch == c], thr, rec[ch == c]) for c in np.unique(ch)}
    return res, d


def table(results, key):
    names = list(results)
    hdr = "| Model | val AUROC | " + " | ".join(f"{k} AUROC" for k in SETS) + " | " + " | ".join(f"{k} ρ" for k in SETS) + " |"
    rows = [hdr, "|" + "---|" * (2 + 2 * len(SETS))]
    for n in names:
        r = results[n]
        a = " | ".join(f"{r[k][key]['auroc']:.3f}" for k in SETS)
        b = " | ".join(f"{r[k][key]['burden_spearman']:.2f}" for k in SETS)
        rows.append(f"| {n} | {r['val_auroc']:.3f} | {a} | {b} |")
    return "\n".join(rows)


def plots(name, d, res):
    out = OUT / "plots"
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for k in SETS:
        fpr, tpr, _ = roc_curve(d[f"{k}__y"], d[f"{k}__s"])
        ax.plot(fpr, tpr, label=f"{k} (AUC {res[k]['all']['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
    ax.set_title(f"ROC - {name}"); ax.legend(); fig.tight_layout(); fig.savefig(out / f"roc_{name}.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, k in zip(axs, SETS):
        m = res[k]["all"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"]); ax.set_title(k)
    fig.tight_layout(); fig.savefig(out / f"confusion_{name}.png", dpi=150); plt.close(fig)

    thr = res["threshold"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, k in zip(axs, SETS):
        y, s, rec = d[f"{k}__y"], d[f"{k}__s"], d[f"{k}__rec"]
        rs = np.unique(rec)
        tb = [y[rec == r].mean() for r in rs]; pb = [(s[rec == r] >= thr).mean() for r in rs]
        ax.scatter(tb, pb, s=14); ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("true burden"); ax.set_ylabel("predicted burden")
        ax.set_title(f"{k}  ρ={res[k]['all']['burden_spearman']:.2f}")
    fig.tight_layout(); fig.savefig(out / f"burden_{name}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    tag = sys.argv[1]
    winner = sys.argv[2] if len(sys.argv) > 2 else None
    results = {}
    for p in sorted(SC.glob(f"*_{tag}.npz")):
        results[p.stem], d = evaluate(p)
        if p.stem == winner:
            plots(p.stem, d, results[p.stem])
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(results, indent=1))
    print("ALL WINDOWS\n" + table(results, "all"))
    print("\nSLEEP ONLY\n" + table(results, "sleep_only"))
