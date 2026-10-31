"""Experiments on the context model F. New file; train_new.py / models_new.py are untouched.

  .venv/bin/python exp.py --name F3 --K 1 --patients 200 --epochs 3
  .venv/bin/python exp.py --name F5gru --K 2 --comb gru --aug --aux

K        windows of context on each side (F is K=1: prev, cur, next = 90 s)
comb     mlp (concatenate window embeddings, as F) | gru (BiGRU over the window embeddings)
aug      random polarity flip / amplitude scale / noise, same for every window of a sample
aux      auxiliary sleep-stage loss on the centre window (helps the encoder tell wake from sleep)
Output: results_exp/<name>.json (history) and results_exp/<name>.pt (best val AUROC checkpoint).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import common as C
from models_new import ResEncoder

OUT = Path("results_exp")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
STAGES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


class Ctx(nn.Module):
    """Shared ResNet encoder on 2K+1 windows -> combine -> one apnea logit (and optional stage logits)."""

    def __init__(self, K=1, comb="mlp", aux=False, hidden=128):
        super().__init__()
        self.K, self.comb_kind, self.W = K, comb, 2 * K + 1
        self.enc = ResEncoder()
        if comb == "mlp":
            self.comb = nn.Sequential(nn.Linear(self.W * 128, hidden), nn.ReLU(), nn.Dropout(0.2))
        else:
            self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
            self.comb = nn.Sequential(nn.Dropout(0.2), nn.Linear(128 + 128, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, 1)
        self.stage = nn.Linear(128, 5) if aux else None  # from the centre window's own embedding

    def encode(self, x):  # [B,1,T] -> [B,128]
        return self.enc(x)

    def combine(self, e):  # [B,W,128] -> logit [B], stage logits [B,5] or None
        if self.comb_kind == "mlp":
            h = self.comb(e.reshape(len(e), -1))
        else:
            g, _ = self.gru(e)
            h = self.comb(torch.cat([g[:, self.K], e[:, self.K]], 1))
        centre = e[:, self.K]
        return self.out(h).squeeze(-1), (self.stage(centre) if self.stage is not None else None)

    def forward(self, x):  # x [B,W,T]
        b = x.shape[0]
        return self.combine(self.encode(x.reshape(b * self.W, 1, -1)).reshape(b, self.W, -1))


def pointers(ws, K):
    """Row ids of the window at offsets -K..K for every window (a window with no neighbour points at itself)."""
    offs = []
    for sign, step in ((-1, ws.prev), (1, ws.next)):
        cur, chain = np.arange(len(ws.y)), []
        for _ in range(K):
            cur = step[cur]
            chain.append(cur)
        offs.append(chain)
    return np.stack(offs[0][::-1] + [np.arange(len(ws.y))] + offs[1], 1)  # [N, 2K+1]


def load_train(meta, X_all, patients, K):
    keep = C.train_rows(meta, patients)
    n = len(meta["y"])
    rows = np.unique(np.concatenate([(keep + d).clip(0, n - 1) for d in range(-K, K + 1)]))
    ws = C.gather(X_all, meta, rows)
    return ws, np.searchsorted(rows, keep)


@torch.no_grad()
def predict(model, ws, ptr, bs=512):
    """Score every window. Each window is encoded ONCE, then combined with its neighbours' stored embeddings;
    in eval mode this equals running the full model on every (prev, cur, next) stack, at 1/(2K+1) of the cost."""
    model.eval()
    emb = []
    for a in range(0, len(ws.y), bs * 2):
        x = torch.from_numpy(ws.X[a:a + bs * 2]).to(DEV).float()[:, None]
        emb.append(model.encode(x))
    emb = torch.cat(emb)
    ptr_t = torch.from_numpy(ptr).to(DEV)
    out = [model.combine(emb[ptr_t[a:a + 8192]])[0] for a in range(0, len(ws.y), 8192)]
    return torch.cat(out).float().cpu().numpy()


def augment(x):
    """x [B,W,T]: one random polarity, gain and noise level per sample, shared by its windows."""
    b = x.shape[0]
    sign = torch.where(torch.rand(b, 1, 1, device=x.device) < 0.5, -1.0, 1.0)
    gain = 0.8 + 0.4 * torch.rand(b, 1, 1, device=x.device)
    return x * sign * gain + 0.05 * torch.randn_like(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--comb", default="mlp", choices=["mlp", "gru"])
    p.add_argument("--aug", action="store_true")
    p.add_argument("--aux", action="store_true")
    p.add_argument("--patients", type=int, default=200)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aux_w", type=float, default=0.2, help="weight of the stage loss")
    p.add_argument("--init", default=None, help="checkpoint to start from (same architecture)")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)

    meta, X_all = C.load_meta(), C.open_all()
    val = C.gather(X_all, meta, C.split_rows(meta, 1))
    vptr = pointers(val, a.K)
    tr, samp = load_train(meta, X_all, a.patients, a.K)
    tptr = pointers(tr, a.K)
    stage_id = np.array([STAGES.get(s, 0) for s in tr.stage])
    print(f"{a.name}: train samples {len(samp)} ({100 * tr.y[samp].mean():.1f}% pos), val {len(val.y)}", flush=True)

    torch.manual_seed(a.seed)
    model = Ctx(a.K, a.comb, a.aux).to(DEV)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=DEV))
    steps = a.epochs * (len(samp) // a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(a.seed)
    best, bad, hist = -1, 0, []
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(samp)
        t0, tot, nb = time.time(), 0.0, len(order) // a.bs
        for b in range(nb):
            idx = order[b * a.bs:(b + 1) * a.bs]
            x = torch.from_numpy(tr.X[tptr[idx]]).to(DEV).float()  # [B,W,T]
            y = torch.from_numpy(tr.y[idx]).to(DEV).float()
            if a.aug:
                x = augment(x)
            logit, st = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if st is not None:
                loss = loss + a.aux_w * F.cross_entropy(st, torch.from_numpy(stage_id[idx]).to(DEV))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item()
            if b % 200 == 0:
                print(f"  ep{ep} {b}/{nb} loss {tot / (b + 1):.4f}", flush=True)
        auc = roc_auc_score(val.y, predict(model, val, vptr))
        hist.append(dict(epoch=ep, train_loss=tot / nb, val_auroc=float(auc), sec=time.time() - t0))
        print(f"epoch {ep}: loss {tot / nb:.4f} val AUROC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best, bad = auc, 0
            torch.save(model.state_dict(), OUT / f"{a.name}.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break
    (OUT / f"{a.name}.json").write_text(json.dumps(dict(args=vars(a), best_val_auroc=float(best), history=hist), indent=1))
    print(f"DONE {a.name} best val AUROC {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
