"""Train one architecture on the balanced CinC cache.  e.g.  .venv/bin/python train_new.py --arch B --patients 200 --epochs 3 --tag screen
Arch E = band-power features + logistic regression / gradient boosting (non-deep baseline)."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
from models_new import CONTEXT, make_model

OUT = Path("results_final/models")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def predict(model, ws, ctx, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(ws.y), bs):
            idx = np.arange(a, min(a + bs, len(ws.y)))
            x = torch.from_numpy(ws.batch(idx, ctx)).to(DEV).float()
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def load_train(meta, X_all, patients, ctx):
    keep = C.train_rows(meta, patients)
    rows = np.unique(np.concatenate([keep, keep - 1, keep + 1]).clip(0, len(meta["y"]) - 1)) if ctx else keep
    ws = C.gather(X_all, meta, rows)
    return ws, np.searchsorted(rows, keep)


def train_deep(a, meta, X_all, val):
    ctx = a.arch in CONTEXT
    tr, samp = load_train(meta, X_all, a.patients, ctx)
    print(f"train samples {len(samp)} ({100 * tr.y[samp].mean():.1f}% pos), val {len(val.y)}", flush=True)
    torch.manual_seed(0)
    model = make_model(a.arch).to(DEV)
    steps = a.epochs * (len(samp) // a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(0)
    best, bad, hist = -1, 0, []
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(samp)
        t0, tot = time.time(), 0.0
        nb = len(order) // a.bs
        for b in range(nb):
            idx = order[b * a.bs:(b + 1) * a.bs]
            x = torch.from_numpy(tr.batch(idx, ctx)).to(DEV).float()
            y = torch.from_numpy(tr.y[idx]).to(DEV).float()
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item()
            if b % 200 == 0:
                print(f"  ep{ep} {b}/{nb} loss {tot / (b + 1):.4f}", flush=True)
        auc = roc_auc_score(val.y, predict(model, val, ctx))
        hist.append(dict(epoch=ep, train_loss=tot / nb, val_auroc=float(auc), sec=time.time() - t0))
        print(f"epoch {ep}: loss {tot / nb:.4f} val AUROC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best, bad = auc, 0
            torch.save(model.state_dict(), OUT / f"{a.arch}_{a.tag}.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break
    return dict(arch=a.arch, tag=a.tag, best_val_auroc=float(best), history=hist)


def train_feat(a, meta, X_all, val):
    tr, samp = load_train(meta, X_all, a.patients, False)
    Ftr, Fva = C.features(tr.X), C.features(val.X)
    sc = StandardScaler().fit(Ftr)
    lr = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(Ftr), tr.y)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, early_stopping=False, random_state=0).fit(Ftr, tr.y)
    res = {}
    for name, m, f in (("E_lr", lr, sc.transform(Fva)), ("E_gbm", gbm, Fva)):
        res[name] = float(roc_auc_score(val.y, m.decision_function(f) if name == "E_lr" else m.predict_proba(f)[:, 1]))
        print(name, "val AUROC", res[name], flush=True)
    joblib.dump(dict(scaler=sc, lr=lr, gbm=gbm), OUT / f"E_{a.tag}.joblib")
    return dict(arch="E", tag=a.tag, best_val_auroc=max(res.values()), per_model=res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--patients", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--tag", default="screen")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--patience", type=int, default=2)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, X_all = C.load_meta(), C.open_all()
    val = C.gather(X_all, meta, C.split_rows(meta, 1))
    res = (train_feat if a.arch == "E" else train_deep)(a, meta, X_all, val)
    (OUT / f"{a.arch}_{a.tag}.json").write_text(json.dumps(res, indent=1))
