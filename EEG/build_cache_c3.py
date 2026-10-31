"""Second EEG channel (C3-M2) for the same cleaned CinC windows, row-aligned with data/cache/meta.npz.

Output: data/cache/X2_all.f16 (float16 [n,3750], same row order as X_all.f16).
"""
from pathlib import Path

import numpy as np

SRC, OUT = Path("data/processed/cinc2018"), Path("data/cache")

if __name__ == "__main__":
    m = np.load(OUT / "meta.npz")
    rec, widx = m["rec"], m["widx"]
    starts = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1]])
    ends = np.r_[starts[1:], len(rec)]
    with open(OUT / "X2_all.f16", "wb") as f:
        for k, (a, b) in enumerate(zip(starts, ends)):
            X2 = np.load(SRC / f"{rec[a]}.npz")["X_c3"][widx[a:b]]
            f.write(np.where(np.isfinite(X2), X2, 0).astype(np.float16).tobytes())
            if k % 100 == 0:
                print(f"{k}/{len(starts)} {rec[a]}", flush=True)
    print("rows", len(rec), "bytes", (OUT / "X2_all.f16").stat().st_size, "expected", len(rec) * 3750 * 2)
