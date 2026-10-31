"""Preprocess the three external test sets exactly like CinC (42 Hz low-pass, 125 Hz, 30 s windows, per-window z-score).

Output: data/processed/external/<set>_<rec>.npz with X float32 [n,3750], y int64, stage <U2, plus channel/fs metadata.
Reuses the helper functions from preprocess.py (filter / resample / segment / z-score / UCD event parsing / labels).
"""
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pyedflib
import wfdb

import preprocess as P

RAW = Path("data/raw")
OUT = Path("data/processed/external")
WIN = P.WINDOW_SAMPLES

# UCDDB _stage.txt: one value per 30 s epoch. 0=wake, 1=REM, 2=S1, 3=S2, 4=S3, 5=S4 (PhysioNet docs)
UCD_STAGE = {"0": "W", "1": "R", "2": "N1", "3": "N2", "4": "N3", "5": "N3"}
# DREAMS hypnogram, 5 s epochs: 5=wake, 4=REM, 3=S1, 2=S2, 1=S3, 0=S4, negative/other = unscored
DREAMS_STAGE = {5: "W", 4: "R", 3: "N1", 2: "N2", 1: "N3", 0: "N3"}
MIT_APNEA = {"H", "HA", "OA", "X", "CA", "CAA"}


def to_windows(sig, fs):
    sig = P.lowpass_filter(sig.astype(np.float64), fs)
    sig = P.resample_signal(sig, fs)
    return P.zscore_segments(P.segment_signal(sig))


def save(name, X, y, stage, **meta):
    n = min(len(X), len(y), len(stage))
    X, y, stage = X[:n], y[:n], np.array(stage[:n], dtype="<U2")
    np.savez_compressed(OUT / f"{name}.npz", X=X, y=y.astype(np.int64), stage=stage, **meta)
    print(f"{name}: {n} windows, {int(y[:n].sum())} pos ({100 * y[:n].mean():.1f}%), stages {dict(Counter(stage))}")


def ucd():
    subjects = sorted({m.group(1) for m in re.finditer(r"ucddb(\d+)\.rec", (RAW / "ucd/RECORDS").read_text())})
    print("UCD subjects:", len(subjects))
    for s in subjects:
        r = pyedflib.EdfReader(str(RAW / f"ucd/ucddb{s}.rec"))
        labels = r.getSignalLabels()
        idx = next(i for i, l in enumerate(labels) if P.normalize_channel_name(l) == "C4A1")
        fs, start = r.getSampleFrequency(idx), r.getStartdatetime()
        eeg = r.readSignal(idx)
        r.close()
        events = P.read_respiratory_events(RAW / f"ucd/ucddb{s}_respevt.txt", start)
        X = to_windows(eeg, fs)
        y = P.create_labels(len(X), events)
        st = [UCD_STAGE.get(l.strip(), "") for l in (RAW / f"ucd/ucddb{s}_stage.txt").read_text().split()]
        st += [""] * (len(X) - len(st))  # a few records have 1 fewer stage than windows
        save(f"ucddb_{s}", X, y, st, channel="C4A1", fs=fs)


def mitbih():
    for rec in (RAW / "mitbih_slpdb/RECORDS").read_text().split():
        path = str(RAW / "mitbih_slpdb" / rec)
        h = wfdb.rdheader(path)
        ch = next(i for i, n in enumerate(h.sig_name) if n.startswith("EEG"))
        sig = wfdb.rdrecord(path, channels=[ch]).p_signal[:, 0]
        notes = wfdb.rdann(path, "st").aux_note
        X = to_windows(np.nan_to_num(sig), h.fs)
        n = min(len(X), len(notes))
        toks = [nt.replace("\x00", " ").split() for nt in notes[:n]]  # slp41/slp45 notes carry trailing NULs
        y = np.array([int(any(t in MIT_APNEA for t in ts)) for ts in toks])
        stage = [{"1": "N1", "2": "N2", "3": "N3", "4": "N3", "R": "R", "W": "W"}.get(ts[0] if ts else "", "") for ts in toks]
        save(f"mitbih_{rec}", X[:n], y, stage, channel=h.sig_name[ch], fs=h.fs)


def read_edf_channel(path, wanted):
    """Minimal EDF reader (pyedflib rejects the DREAMS files). Returns (signal, fs). Tolerates a wrong record count."""
    raw = Path(path).read_bytes()
    ns, rec_dur = int(raw[252:256]), float(raw[244:252])
    hdr = raw[256:256 + ns * 256]
    off, f = 0, {}
    for name, w in [("label", 16), ("trans", 80), ("dim", 8), ("pmin", 8), ("pmax", 8), ("dmin", 8), ("dmax", 8), ("pre", 80), ("n", 8), ("res", 32)]:
        f[name] = [hdr[off + i * w:off + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]
        off += ns * w
    nsamp = [int(v) for v in f["n"]]
    data = np.frombuffer(raw[256 + ns * 256:], dtype="<i2")
    nrec = len(data) // sum(nsamp)
    data = data[:nrec * sum(nsamp)].reshape(nrec, sum(nsamp))
    i = f["label"].index(wanted)
    a = sum(nsamp[:i])
    return data[:, a:a + nsamp[i]].reshape(-1).astype(np.float64), nsamp[i] / rec_dur


def read_dreams_events(path):
    events, active = [], False
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if line.startswith("["):
            active = any(k in line for k in ("hypopnea", "obstructive_apnea", "central_apnea", "mixed_apnea"))
            continue
        parts = line.split()
        if active and len(parts) == 2:
            events.append({"start": float(parts[0]), "duration": float(parts[1]), "type": "x"})
    return events


def dreams():
    for k in range(1, 13):
        d = RAW / "dreams_apnea"
        eeg, fs = read_edf_channel(d / f"excerpt{k}.edf", "CZ-A1")
        X = to_windows(eeg, fs)
        y = P.create_labels(len(X), read_dreams_events(d / f"Visual_scoring1_excerpt{k}.txt"))
        hyp = [int(float(t)) for t in (d / f"Hypnogram_excerpt{k}.txt").read_text().split() if not t.startswith("[")]
        stage = []
        for w in range(len(X)):  # 6 five-second epochs per 30 s window -> majority vote
            votes = [DREAMS_STAGE.get(v, "") for v in hyp[6 * w:6 * w + 6]]
            stage.append(Counter(votes).most_common(1)[0][0] if votes else "")
        save(f"dreams_{k}", X, y, stage, channel="CZ-A1", fs=fs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowpass", type=float, default=P.LOWPASS_CUTOFF, help="low-pass cutoff in Hz (default 42, as in CinC)")
    ap.add_argument("--out", default=str(OUT), help="output folder")
    a = ap.parse_args()
    P.LOWPASS_CUTOFF, OUT = a.lowpass, Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    ucd()
    mitbih()
    dreams()
