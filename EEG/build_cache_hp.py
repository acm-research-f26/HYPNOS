"""Write the high-passed (apnea.band) version of both channels to disk, so training can memory-map it.

Usage: python build_cache_hp.py HZ [--causal]  ->  data/cache/X_hp{HZ}[c].f16, X2_hp{HZ}[c].f16
"""
import sys

import numpy as np

import apnea as A

if __name__ == "__main__":
    A.HP, A.CAUSAL = float(sys.argv[1]), "--causal" in sys.argv
    n = len(A.meta()["y"])
    for src, dst in (("X_all", "X"), ("X2_all", "X2")):
        X = np.memmap(A.CACHE / f"{src}.f16", np.float16, "r", shape=(n, A.T))
        out = np.memmap(A.CACHE / A.hp_name(dst), np.float16, "w+", shape=(n, A.T))
        for a in range(0, n, 40000):
            out[a:a + 40000] = A.band(np.ascontiguousarray(X[a:a + 40000]))
            print(dst, a, flush=True)
        out.flush()
