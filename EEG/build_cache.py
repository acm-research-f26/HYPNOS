"""Clean CinC2018 windows and build one float16 memmap of every clean window plus a metadata table.

Cleaning: drop unscored windows (stage==""), flat / near-flat windows (z-scored std<0.5 -> all-zero rows),
non-finite windows, and windows where >50 % of consecutive samples are identical (clipped / stuck).
Patient split 80/10/10 (seed 0). Output in data/cache/: X_all.f16 (raw memmap), meta.npz.
Input: data/processed/cinc2018/<rec>.npz from preprocess_cinc2018.py.
The balanced train subset (all positives + 2x negatives per patient) is drawn by train_rows() in apnea.py from meta.npz.
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SRC = Path("data/processed/cinc2018")
OUT = Path("data/cache")


def load(path):
    d = np.load(path)
    X, y, stage = d["X"], d["y"], d["stage"]
    n0 = len(X)
    finite = np.isfinite(X).all(1)
    Xs = np.where(np.isfinite(X), X, 0)
    flat = Xs.std(1) < 0.5
    stuck = (np.diff(Xs, axis=1) == 0).mean(1) > 0.5
    unscored = stage == ""
    keep = ~(unscored | flat | stuck | ~finite)
    stats = dict(total=n0, unscored=int(unscored.sum()), flat=int((flat & ~unscored).sum()),
                 stuck=int((stuck & ~flat & ~unscored).sum()), nonfinite=int((~finite).sum()))
    return path.stem, Xs[keep].astype(np.float16), y[keep], stage[keep], np.flatnonzero(keep), stats


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.npz"))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(files))
    split_of = {}
    for rank, i in enumerate(perm):
        split_of[files[i].stem] = 0 if rank < 0.8 * len(files) else (1 if rank < 0.9 * len(files) else 2)  # 0 train,1 val,2 test

    recs, widx, ys, stages, splits = [], [], [], [], []
    tot = dict(total=0, unscored=0, flat=0, stuck=0, nonfinite=0)
    with open(OUT / "X_all.f16", "wb") as f, Pool(4) as pool:
        for k, (name, X, y, st, wi, stats) in enumerate(pool.imap(load, files)):
            f.write(X.tobytes())
            recs += [name] * len(X); widx.append(wi); ys.append(y); stages.append(st)
            splits.append(np.full(len(X), split_of[name], np.int8))
            for key in tot:
                tot[key] += stats[key]
            if k % 50 == 0:
                print(f"{k}/{len(files)} {name}", flush=True)
    meta = dict(rec=np.array(recs), widx=np.concatenate(widx), y=np.concatenate(ys), stage=np.concatenate(stages),
                split=np.concatenate(splits))
    np.savez(OUT / "meta.npz", **meta)
    n = len(meta["y"])
    print("dropped:", tot, "kept:", n, "of", tot["total"])
    for s, nm in enumerate(["train", "val", "test"]):
        m = meta["split"] == s
        print(nm, "patients", len(set(meta["rec"][m])), "windows", int(m.sum()), "pos%", round(100 * meta["y"][m].mean(), 2))
