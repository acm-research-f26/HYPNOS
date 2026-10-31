"""Shared data access, batching, features and metrics for the new apnea pipeline."""
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

CACHE = Path("data/cache")
EXT = Path("data/processed/external")
T = 3750


# ---------------------------------------------------------------- data
class WinSet:
    """In-RAM float16 windows + labels + prev/next pointers (local row ids; self if no adjacent window)."""

    def __init__(self, X, y, stage, rec, widx):
        self.X, self.y, self.stage, self.rec, self.widx = X, y, stage, rec, widx
        n = len(y)
        same = lambda a, b: (rec[a] == rec[b]) & (widx[a] == widx[b] + 1)
        i = np.arange(n)
        self.prev = np.where((i > 0) & same(i, np.maximum(i - 1, 0)), i - 1, i)
        self.next = np.where((i < n - 1) & same(np.minimum(i + 1, n - 1), i), i + 1, i)

    def batch(self, idx, ctx):
        if not ctx:
            return self.X[idx][:, None, :]
        return np.stack([self.X[self.prev[idx]], self.X[idx], self.X[self.next[idx]]], 1)


def load_meta():
    m = np.load(CACHE / "meta.npz")
    return {k: m[k] for k in m.files}


def open_all():
    n = len(load_meta()["y"])
    return np.memmap(CACHE / "X_all.f16", dtype=np.float16, mode="r", shape=(n, T))


def gather(X_all, meta, rows):
    """Load `rows` (global ids, any order, no dups) into a WinSet that keeps the global adjacency (rows must be
    contiguous runs or include neighbours if context is wanted)."""
    rows = np.sort(rows)
    return WinSet(np.ascontiguousarray(X_all[rows]), meta["y"][rows], meta["stage"][rows], meta["rec"][rows],
                  meta["widx"][rows])


def split_rows(meta, s):
    return np.flatnonzero(meta["split"] == s)


def train_rows(meta, n_patients=0, neg_ratio=2, seed=0):
    """Balanced training rows: all positives + neg_ratio x as many negatives per patient (optionally only the first
    n_patients patients, in a seeded random order)."""
    rng = np.random.default_rng(seed)
    recs = np.unique(meta["rec"][meta["split"] == 0])
    rng.shuffle(recs)
    if n_patients:
        recs = recs[:n_patients]
    keep = []
    order = np.argsort(meta["rec"], kind="stable")
    sr = meta["rec"][order]
    for r in recs:
        lo, hi = np.searchsorted(sr, r), np.searchsorted(sr, r, side="right")
        rows = order[lo:hi]
        pos, neg = rows[meta["y"][rows] == 1], rows[meta["y"][rows] == 0]
        neg = rng.choice(neg, min(len(neg), neg_ratio * max(len(pos), 20)), replace=False)
        keep.append(np.concatenate([pos, neg]))
    keep = np.concatenate(keep)
    return np.sort(keep)


def load_external():
    """Returns {set: WinSet} for ucddb / mitbih / dreams. Drops unscored + flat/non-finite windows like the CinC clean."""
    out = {}
    for ds in ("ucddb", "mitbih", "dreams"):
        Xs, ys, sts, recs, widx, chans = [], [], [], [], [], []
        for f in sorted(EXT.glob(f"{ds}_*.npz")):
            d = np.load(f)
            X, y, st = d["X"], d["y"], d["stage"]
            keep = (st != "") & np.isfinite(X).all(1) & (X.std(1) > 0.5)
            Xs.append(X[keep].astype(np.float16)); ys.append(y[keep]); sts.append(st[keep])
            recs += [f.stem] * int(keep.sum()); widx.append(np.flatnonzero(keep))
            chans += [str(d["channel"])] * int(keep.sum())
        w = WinSet(np.concatenate(Xs), np.concatenate(ys), np.concatenate(sts), np.array(recs), np.concatenate(widx))
        w.channel = np.array(chans)
        out[ds] = w
    return out


# ---------------------------------------------------------------- features (arch E)
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30), "gamma": (30, 42)}


def features(X, chunk=4000, fs=125):
    """Band-power / spectral / Hjorth / temporal-variability features for z-scored windows [N,3750] -> [N,F]."""
    out = []
    for a in range(0, len(X), chunk):
        x = X[a:a + chunk].astype(np.float32)
        f, p = welch(x, fs=fs, nperseg=250, noverlap=125, axis=-1)  # p [n, 126]; 0.5 Hz bins
        tot = p[:, (f >= 0.5) & (f <= 42)].sum(1, keepdims=True) + 1e-12
        bp = np.stack([p[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in BANDS.values()], 1)
        rel = bp / tot
        pn = p[:, (f >= 0.5) & (f <= 42)] / tot
        ent = -(pn * np.log(pn + 1e-12)).sum(1, keepdims=True)
        cum = np.cumsum(pn, 1)
        sef = np.stack([f[(f >= 0.5) & (f <= 42)][(cum >= q).argmax(1)] for q in (0.5, 0.9, 0.95)], 1)
        d1, d2 = np.diff(x, axis=1), np.diff(x, 2, axis=1)
        mob = d1.std(1) / (x.std(1) + 1e-8)
        comp = (d2.std(1) / (d1.std(1) + 1e-8)) / (mob + 1e-8)
        ll = np.abs(d1).mean(1)
        zc = (np.diff(np.signbit(x), axis=1) != 0).mean(1)
        # temporal structure: 6 sub-windows of 5 s -> spread of power and of delta/alpha ratio over time
        sub = x.reshape(len(x), 6, -1)
        sp = (sub ** 2).mean(2)
        sf, sw = welch(sub, fs=fs, nperseg=250, axis=-1)
        sd = sw[..., (sf >= 0.5) & (sf < 4)].sum(-1)
        sa = sw[..., (sf >= 8) & (sf < 16)].sum(-1)
        ratio = np.log(sd + 1e-9) - np.log(sa + 1e-9)
        out.append(np.concatenate([
            np.log(bp + 1e-9), rel, np.log(tot), ent, sef, mob[:, None], comp[:, None], ll[:, None], zc[:, None],
            kurtosis(x, axis=1)[:, None], skew(x, axis=1)[:, None],
            np.log(sp.std(1, keepdims=True) + 1e-9), np.log(sp.max(1, keepdims=True) + 1e-9),
            ratio.std(1, keepdims=True), ratio.max(1, keepdims=True) - ratio.min(1, keepdims=True),
            np.log(sd + 1e-9).std(1, keepdims=True), np.log(sa + 1e-9).std(1, keepdims=True)], 1))
    return np.nan_to_num(np.concatenate(out).astype(np.float32))


# ---------------------------------------------------------------- metrics
def pick_threshold(y, s):
    """Threshold maximising MCC on validation (scanned over score quantiles)."""
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    best = max(qs, key=lambda t: matthews_corrcoef(y, s >= t))
    return float(best)


def metrics(y, s, thr, rec=None):
    p = s >= thr
    out = dict(n=int(len(y)), prevalence=float(y.mean()), auroc=float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out.update(mcc=float(matthews_corrcoef(y, p)), sensitivity=float(tp / max(tp + fn, 1)),
               specificity=float(tn / max(tn + fp, 1)), f1=float(f1_score(y, p, zero_division=0)),
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    if rec is not None:
        recs = np.unique(rec)
        pred_b = np.array([p[rec == r].mean() for r in recs])
        true_b = np.array([y[rec == r].mean() for r in recs])
        rho = spearmanr(pred_b, true_b)
        out.update(burden_spearman=float(rho.statistic), burden_p=float(rho.pvalue), n_patients=int(len(recs)))
    return out
