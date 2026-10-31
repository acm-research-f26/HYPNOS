"""Plot example 30 s windows (stage x label) from the internal test split."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common as C

meta, X = C.load_meta(), C.open_all()
rng = np.random.default_rng(1)
t = np.arange(C.T) / 125
combos = [("W", 0), ("N1", 1), ("N2", 0), ("N2", 1), ("N3", 0), ("R", 1)]
fig, axs = plt.subplots(len(combos), 1, figsize=(11, 10), sharex=True)
for ax, (st, y) in zip(axs, combos):
    rows = np.flatnonzero((meta["split"] == 2) & (meta["stage"] == st) & (meta["y"] == y))
    r = rng.choice(rows)
    ax.plot(t, X[r].astype(np.float32), lw=0.6, color="C3" if y else "C0")
    ax.set_ylabel(f"{st}\n{'apnea' if y else 'no apnea'}", rotation=0, ha="right", va="center")
    ax.set_ylim(-5, 5); ax.set_title(meta["rec"][r] + f" window {meta['widx'][r]}", fontsize=8, loc="right")
axs[-1].set_xlabel("seconds"); fig.suptitle("Example 30 s C4-M1 windows (z-scored, 125 Hz)")
fig.tight_layout(); fig.savefig("results_final/plots/examples.png", dpi=130)
