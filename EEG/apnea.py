"""Single-channel EEG apnea detection: data, models, training, scoring and reporting in one file.
  python apnea.py train NAME [--aux --epochs 14 --patients 0]    -> results_exp/NAME.pt, NAME.json
  python apnea.py score NAME [NAME ...]                          -> results_exp/scores/NAME.npz (val, test, external)
  python apnea.py report NAME [NAME ...] [--plots FILE.png]      several names = ensemble (mean of val-standardised scores)
Models (all take z-scored 30 s windows, 125 Hz, 0.5-42 Hz): Ctx = ResNet encoder on prev/cur/next window (+ optional
sleep-stage side loss); GRU = CNN + BiGRU on one window. Threshold = max MCC (or max sens+spec) on val, unchanged elsewhere."""
import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

CACHE, EXT, OUT = Path("data/cache"), Path("data/processed/external"), Path("results_exp")
T, SETS = 3750, ["test", "ucddb", "mitbih", "dreams"]
STAGE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
HP = 0.0  # extra high-pass (Hz) applied to every window at load time; set by --hp. 0 keeps the 0.5-42 Hz cache as is.
CAUSAL = False  # --causal: forward-only filter (as on a device) instead of zero-phase


def band(X):
    """Butterworth high-pass at HP Hz (zero-phase, or forward-only with CAUSAL), then z-score each window again (identity when HP is 0)."""
    if not HP:
        return X
    sos, out = butter(4, HP, "highpass", fs=125, output="sos"), np.empty_like(X)
    for a in range(0, len(X), 20000):
        x = (sosfilt if CAUSAL else sosfiltfilt)(sos, X[a:a + 20000].astype(np.float32), axis=1)
        out[a:a + 20000] = ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)).astype(np.float16)
    return out


# ------------------------------------------------------------------ data
class Win:
    """Windows [n,T] float16 + labels; ptr[i] = rows of windows i-K..i+K in the same recording (itself if missing)."""

    def __init__(s, X, y, stage, rec, widx):
        s.X, s.y, s.stage, s.rec, s.widx = X, y, stage, rec, widx
        s.Xs = [X]  # one array per EEG channel; channel 0 is C4-M1

    def ctx(s, K):
        i, n = np.arange(len(s.y)), len(s.y)

        def near(d):
            j = (i + d).clip(0, n - 1)
            return np.where((s.rec[j] == s.rec) & (s.widx[j] == s.widx + d), j, i)

        s.ptr = np.stack([near(d) for d in range(-K, K + 1)], 1)
        return s


def hp_name(prefix):
    """File name of the pre-filtered cache written by build_cache_hp.py."""
    return f"{prefix}_hp{HP:g}{'c' if CAUSAL else ''}.f16"


def meta():
    m = np.load(CACHE / "meta.npz")
    return {k: m[k] for k in m.files}


def get(m, rows, mc=False):
    """Windows for `rows`; with mc also the second channel (C3-M2, from build_cache_c3.py) as w.Xs[1]."""
    X = np.memmap(CACHE / "X_all.f16", np.float16, "r", shape=(len(m["y"]), T))
    w = Win(band(np.ascontiguousarray(X[rows])), *(m[k][rows] for k in ("y", "stage", "rec", "widx")))
    if mc:
        X2 = np.memmap(CACHE / "X2_all.f16", np.float16, "r", shape=(len(m["y"]), T))
        w.Xs.append(band(np.ascontiguousarray(X2[rows])))
    return w


def train_mmap(m, mc):
    """All cleaned windows as memory-mapped pre-filtered arrays: no copy in RAM, so a big training set cannot cause swapping."""
    n = len(m["y"])
    mm = lambda p: np.memmap(CACHE / hp_name(p), np.float16, "r", shape=(n, T))
    w = Win(mm("X"), *(m[k] for k in ("y", "stage", "rec", "widx")))
    if mc:
        w.Xs.append(mm("X2"))
    return w


def train_rows(m, n_patients, neg_ratio=2, seed=0):
    """All positives + neg_ratio x as many negatives per training patient (first n_patients of a seeded shuffle)."""
    rng = np.random.default_rng(seed)
    recs = np.unique(m["rec"][m["split"] == 0])
    rng.shuffle(recs)
    order = np.argsort(m["rec"], kind="stable")
    sr, keep = m["rec"][order], []
    for r in recs[:n_patients or None]:
        rows = order[np.searchsorted(sr, r):np.searchsorted(sr, r, "right")]
        pos, neg = rows[m["y"][rows] == 1], rows[m["y"][rows] == 0]
        keep += [pos, rng.choice(neg, min(len(neg), neg_ratio * max(len(pos), 20)), replace=False)]
    return np.sort(np.concatenate(keep))


def external():
    """UCDDB / MIT-BIH / DREAMS windows, dropping unscored, flat and non-finite ones like the CinC cleaning."""
    out = {}
    for ds in SETS[1:]:
        cols = []
        for f in sorted(EXT.glob(f"{ds}_*.npz")):
            d = np.load(f)
            X = d["X"]
            ok = (d["stage"] != "") & np.isfinite(X).all(1) & (X.std(1) > 0.5)
            cols.append((band(X[ok].astype(np.float16)), d["y"][ok], d["stage"][ok], np.full(ok.sum(), f.stem),
                         np.flatnonzero(ok), np.full(ok.sum(), str(d["channel"]))))
        cols = [np.concatenate(c) for c in zip(*cols)]
        out[ds] = Win(*cols[:5])
        out[ds].channel = cols[5]
    return out


# ------------------------------------------------------------------ models
class Block(nn.Module):
    def __init__(self, cin, cout, stride, k=7):
        super().__init__()
        self.c1, self.b1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm1d(cout)
        self.c2, self.b2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False), nn.BatchNorm1d(cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        h = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(h + (x if self.skip is None else self.skip(x)))


class ResEncoder(nn.Module):
    """1D ResNet: window [B,1,T] -> 128 numbers (average + max pooled over time)."""

    def __init__(self, width=(16, 32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, width[0], 15, 2, 7, bias=False), nn.BatchNorm1d(width[0]), nn.ReLU(),
                                  nn.MaxPool1d(2))
        blocks, c = [], width[0]
        for w in width:
            blocks += [Block(c, w, 2 if w != width[0] else 1), Block(w, w, 1)]
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * c, 128), nn.ReLU())

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.proj(torch.cat([h.mean(-1), h.amax(-1)], 1))


class Ctx(nn.Module):
    """Shared encoder on the 2K+1 windows around a window -> MLP -> apnea logit (+ stage logits from the centre)."""

    def __init__(self, K=1, aux=False):
        super().__init__()
        self.K, self.enc = K, ResEncoder()
        self.comb = nn.Sequential(nn.Linear((2 * K + 1) * 128, 128), nn.ReLU(), nn.Dropout(0.2))
        self.out = nn.Linear(128, 1)
        self.stage = nn.Linear(128, 5) if aux else None

    def encode(self, x):  # [B,1,T] -> [B,128]
        return self.enc(x)

    def combine(self, e):  # [B,2K+1,128] -> logit [B], stage logits [B,5] or None
        return self.out(self.comb(e.flatten(1))).squeeze(-1), (self.stage(e[:, self.K]) if self.stage else None)

    def forward(self, x):  # [B,2K+1,T]
        b = len(x)
        return self.combine(self.encode(x.reshape(b * x.shape[1], 1, -1)).reshape(b, x.shape[1], -1))


class MC(Ctx):
    """Channel-flexible Ctx: the shared encoder runs on every channel and a masked mean over channels feeds the same head,
    so any number of channels works. Training drops one channel at random (channel dropout)."""

    def __init__(self, K=1, aux=False):
        super().__init__(K, aux)
        self.register_buffer("mc", torch.ones(1))  # marks the checkpoint as channel-flexible

    def encode(self, x):  # [B,C,T] -> [B,128]
        b, c, t = x.shape
        return self.enc(x.reshape(b * c, 1, t)).reshape(b, c, -1).mean(1)

    def forward(self, x, mask):  # x [B,C,W,T], mask [B,C] (1 = channel present)
        b, c, w, t = x.shape
        mk = mask[:, :, None, None]
        e = self.enc(x.reshape(b * c * w, 1, t)).reshape(b, c, w, -1)
        return self.combine((e * mk).sum(1) / mk.sum(1))


class GRU(nn.Module):
    """CNN front end + BiGRU over time on a single window (model C)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 11, 2, 5, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 7, 1, 3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 5, 1, 2, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.gru = nn.GRU(64, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(4 * hidden, 1))

    def forward(self, x):
        h, _ = self.gru(self.cnn(x).transpose(1, 2))
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1)).squeeze(-1)


def find(name, ext):
    """Newest-first search for a checkpoint (.pt) or saved score file (.npz) by name."""
    dirs = ("results_exp", "results_exp/scores", "results_final/models", "results_final/scores")
    return next(p for d in dirs if (p := Path(d) / f"{name}.{ext}").exists())


def load(name):
    """Rebuild a model from its checkpoint. Old ContextNet files (head.0 / head.3) are renamed to comb.0 / out."""
    sd = torch.load(find(name, "pt"), map_location=DEV)
    if "gru.weight_ih_l0" in sd:
        m = GRU()
    else:
        sd = {k.replace("head.0.", "comb.0.").replace("head.3.", "out."): v for k, v in sd.items()}
        m = (MC if "mc" in sd else Ctx)((sd["comb.0.weight"].shape[1] // 128 - 1) // 2, "stage.weight" in sd)
    m.load_state_dict(sd)
    return m.to(DEV)


@torch.no_grad()
def score(model, w, bs=1024, chans=None):
    """Logit for every window. A context model encodes each window once and reuses it for its neighbours.
    A channel-flexible model uses the channels in `chans` (default: all that the windows have)."""
    model.eval()
    cs = [w.Xs[c] for c in (chans or range(len(w.Xs))) if c < len(w.Xs)] if isinstance(model, MC) else [w.X]
    if not cs:  # none of the requested channels exist in this set
        return np.full(len(w.y), np.nan, np.float32)
    x = lambda a: torch.from_numpy(np.stack([X[a:a + bs] for X in cs], 1)).to(DEV).float()
    n = range(0, len(w.y), bs)
    if not hasattr(model, "encode"):
        return torch.cat([model(x(a)) for a in n]).cpu().numpy()
    e, p = torch.cat([model.encode(x(a)) for a in n]), torch.from_numpy(w.ctx(model.K).ptr).to(DEV)
    return torch.cat([model.combine(e[p[a:a + 8192]])[0] for a in range(0, len(p), 8192)]).cpu().numpy()


# ------------------------------------------------------------------ training
def batches(tr, order, bs, stage):
    """Build batches on a worker thread so the GPU never waits for numpy indexing."""
    q = queue.Queue(4)

    def work():
        for b in range(len(order) // bs):
            i = order[b * bs:(b + 1) * bs]
            x = tr.X[tr.ptr[i]] if len(tr.Xs) == 1 else np.stack([X[tr.ptr[i]] for X in tr.Xs], 1)
            q.put([torch.from_numpy(a) for a in (x, tr.y[i], stage[i])])
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (b := q.get()) is not None:
        yield b


def train(a):
    m, K = meta(), a.K
    val = get(m, np.flatnonzero(m["split"] == 1), a.mc)
    keep = train_rows(m, a.patients)
    rows = np.unique(np.concatenate([(keep + d).clip(0, len(m["y"]) - 1) for d in range(-K, K + 1)]))
    if a.mmap:  # needs build_cache_hp.py output; window indices are global rows
        tr, samp = train_mmap(m, a.mc).ctx(K), keep
    else:
        tr, samp = get(m, rows, a.mc).ctx(K), np.searchsorted(rows, keep)
    stage = np.array([STAGE.get(s, 0) for s in tr.stage])
    print(f"{a.name}: {len(samp)} train samples ({100 * tr.y[samp].mean():.1f}% pos), {len(val.y)} val", flush=True)

    torch.manual_seed(a.seed)
    model = (MC if a.mc else Ctx)(K, a.aux).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * (len(samp) // a.bs), pct_start=0.15)
    rng, best, bad, hist = np.random.default_rng(a.seed), -1, 0, []
    OUT.mkdir(exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, len(samp) // a.bs
        for b, (x, y, st) in enumerate(batches(tr, rng.permutation(samp), a.bs, stage)):
            with torch.autocast(DEV, torch.float16, enabled=a.amp):
                if a.mc:  # channel dropout: with prob chdrop one random channel is hidden
                    mask = torch.ones(len(x), x.shape[1])
                    hide = torch.from_numpy(rng.random(len(x)) < a.chdrop)
                    mask[hide, torch.from_numpy(rng.integers(0, x.shape[1], len(x)))[hide]] = 0
                    logit, st_logit = model(x.to(DEV).float(), mask.to(DEV))
                else:
                    logit, st_logit = model(x.to(DEV).float())
            loss = F.binary_cross_entropy_with_logits(logit.float(), y.to(DEV).float())
            if st_logit is not None:
                loss = loss + a.aux_w * F.cross_entropy(st_logit.float(), st.to(DEV))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item()
            if b % 200 == 0:
                print(f"  ep{ep} {b}/{nb} loss {tot / (b + 1):.4f}", flush=True)
        auc = roc_auc_score(val.y, score(model, val))
        hist.append(dict(epoch=ep, train_loss=tot / nb, val_auroc=float(auc), sec=time.time() - t0))
        print(f"epoch {ep}: loss {tot / nb:.4f} val AUROC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best, bad = auc, 0
            torch.save(model.state_dict(), OUT / f"{a.name}.pt")
        elif (bad := bad + 1) >= a.patience:
            print("early stop"); break
    (OUT / f"{a.name}.json").write_text(json.dumps(dict(args=vars(a), best_val_auroc=float(best), history=hist), indent=1))
    print(f"DONE {a.name} best val AUROC {best:.4f}")


# ------------------------------------------------------------------ scoring and metrics
def score_all(names, suffix="", mc=False, chans=None):
    m = meta()
    sets = {"val": get(m, np.flatnonzero(m["split"] == 1), mc), "test": get(m, np.flatnonzero(m["split"] == 2), mc), **external()}
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    for name in names:
        model, arrs = load(name), {}
        for k, w in sets.items():
            arrs |= {f"{k}__y": w.y, f"{k}__s": score(model, w, chans=chans), f"{k}__rec": w.rec, f"{k}__stage": w.stage}
            if hasattr(w, "channel"):
                arrs[f"{k}__channel"] = w.channel
        np.savez(OUT / "scores" / f"{name}{suffix}.npz", **arrs)
        print("saved", name, flush=True)


def threshold(y, s, rule):
    """mcc: the score that maximises MCC on val. bal: maximises sensitivity + specificity (prevalence-free)."""
    if rule == "bal":
        fpr, tpr, thr = roc_curve(y, s)
        return float(thr[np.argmax(tpr - fpr)])
    ts, ps, ns = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197))), np.sort(s[y == 1]), np.sort(s[y == 0])
    tp, fp = len(ps) - np.searchsorted(ps, ts), len(ns) - np.searchsorted(ns, ts)
    fn, tn = len(ps) - tp, len(ns) - fp
    den = np.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).astype(float))
    return float(ts[np.argmax(np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0))])


def metrics(y, s, t, rec):
    p = s >= t
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((y == 1).sum()) - tp, int((y == 0).sum()) - fp
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    _, g = np.unique(rec, return_inverse=True)
    per = np.bincount(g)
    rho = spearmanr(np.bincount(g, p) / per, np.bincount(g, y) / per).statistic
    return dict(n=len(y), patients=len(per), prevalence=float(y.mean()), auroc=float(roc_auc_score(y, s)),
                mcc=(tp * tn - fp * fn) / den if den else 0.0, sensitivity=tp / max(tp + fn, 1),
                specificity=tn / max(tn + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1), burden_rho=float(rho),
                tp=tp, fp=fp, fn=fn, tn=tn)


def report(names, plots=None):
    z = [np.load(find(n, "npz")) for n in names]
    ks = ["val"] + SETS
    s = {k: np.mean([(d[f"{k}__s"] - d["val__s"].mean()) / d["val__s"].std() for d in z], 0) for k in ks}
    y, rec, st = ({k: z[0][f"{k}__{f}"] for k in ks} for f in ("y", "rec", "stage"))
    for rule in ("mcc", "bal"):
        t, res = threshold(y["val"], s["val"], rule), {}
        print(f"\n{'+'.join(names)}, threshold = {rule} on val\n| set | patients | prevalence | AUROC | MCC | sens | spec | F1 | "
              f"burden ρ | sleep-only AUROC |\n|---|---|---|---|---|---|---|---|---|---|")
        for k in SETS:
            a, sl = metrics(y[k], s[k], t, rec[k]), metrics(y[k][st[k] != "W"], s[k][st[k] != "W"], t, rec[k][st[k] != "W"])
            res[k] = a
            print(f"| {k} | {a['patients']} | {100 * a['prevalence']:.1f}% | {a['auroc']:.3f} | {a['mcc']:.3f} | "
                  f"{a['sensitivity']:.2f} | {a['specificity']:.2f} | {a['f1']:.2f} | {a['burden_rho']:.2f} | {sl['auroc']:.3f} |")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t, (fig, ax) = threshold(y["val"], s["val"], "mcc"), plt.subplots(3, 4, figsize=(15, 11))
        for j, k in enumerate(SETS):
            fpr, tpr, _ = roc_curve(y[k], s[k])
            ax[0, j].plot(fpr, tpr); ax[0, j].plot([0, 1], [0, 1], "k:"); ax[0, j].set_title(f"{k} ROC (AUC {roc_auc_score(y[k], s[k]):.3f})")
            m = metrics(y[k], s[k], t, rec[k])
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            ax[1, j].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (r, c), v in np.ndenumerate(cm):
                ax[1, j].text(c, r, v, ha="center", va="center")
            ax[1, j].set_xticks([0, 1], ["pred 0", "pred 1"]); ax[1, j].set_yticks([0, 1], ["true 0", "true 1"])
            rs = np.unique(rec[k])
            ax[2, j].scatter([y[k][rec[k] == r].mean() for r in rs], [(s[k][rec[k] == r] >= t).mean() for r in rs], s=14)
            ax[2, j].plot([0, 1], [0, 1], "k:"); ax[2, j].set_title(f"burden ρ={m['burden_rho']:.2f}")
            ax[2, j].set_xlabel("true burden"); ax[2, j].set_ylabel("predicted burden")
        fig.tight_layout(); fig.savefig(plots, dpi=130)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("name")
    for flag, typ, default in (("K", int, 1), ("patients", int, 0), ("epochs", int, 14), ("lr", float, 2e-3), ("bs", int, 256),
                               ("patience", int, 3), ("seed", int, 0), ("aux_w", float, 0.2)):
        t.add_argument(f"--{flag}", type=typ, default=default)
    t.add_argument("--aux", action="store_true")
    t.add_argument("--hp", type=float, default=0.0, help="extra high-pass cutoff in Hz")
    t.add_argument("--amp", action="store_true", help="fp16 autocast (about 1.3x faster steps)")
    t.add_argument("--causal", action="store_true", help="forward-only high-pass")
    t.add_argument("--mmap", action="store_true", help="train from the memory-mapped cache of build_cache_hp.py (low RAM)")
    t.add_argument("--mc", action="store_true", help="two-channel (C4-M1 + C3-M2) channel-flexible model")
    t.add_argument("--chdrop", type=float, default=0.3, help="with --mc: probability of hiding one channel per sample")
    sc, rp = sub.add_parser("score"), sub.add_parser("report")
    sc.add_argument("--causal", action="store_true", help="forward-only high-pass")
    sc.add_argument("--ext", help="folder with external-set windows (default data/processed/external)")
    sc.add_argument("--mc", action="store_true", help="load the second channel (needed for channel-flexible models)")
    sc.add_argument("--chans", help="comma-separated channel indices to use, e.g. 0 or 0,1 (default: all)")
    sc.add_argument("--suffix", default="", help="appended to the saved score file name")
    sc.add_argument("names", nargs="+")
    sc.add_argument("--hp", type=float, default=0.0, help="must match the --hp the model was trained with")
    rp.add_argument("names", nargs="+")
    rp.add_argument("--plots")
    a = p.parse_args()
    HP, CAUSAL = getattr(a, "hp", 0.0), getattr(a, "causal", False)
    EXT = Path(getattr(a, "ext", None) or EXT)
    {"train": lambda: train(a), "score": lambda: score_all(a.names, a.suffix, a.mc, [int(c) for c in a.chans.split(",")] if a.chans else None), "report": lambda: report(a.names, a.plots)}[a.cmd]()
