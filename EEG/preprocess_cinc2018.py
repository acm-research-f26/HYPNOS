"""
HYPNOS: stream the PhysioNet 2018 Challenge ("You Snooze You Win") training set
into the same 30 s EEG window format as preprocess.py, deleting raw files as it goes.

For each of the 994 training records:
  1. download only what's needed (.hea, .mat signals, .arousal annotations)
  2. read C4-M1 (+ C3-M2), low-pass 42 Hz, resample 200 -> 125 Hz, cut 30 s windows, z-score
  3. label a window 1 if an apnea/hypopnea overlaps it for >= 10 s (same rule as UCDDB)
  4. save data/processed/cinc2018/<record>.npz, then delete the raw files

Output keys (X / y match the UCDDB .npz files):
  X      float32 [n, 3750]  C4-M1, z-scored per window   (same channel as UCDDB)
  X_c3   float16 [n, 3750]  C3-M2, z-scored per window   (for the 2-channel model)
  y      int64   [n]        1 = apnea/hypopnea window
  stage  <U2     [n]        W / N1 / N2 / N3 / R / "" (unscored)  (for sleep staging)
  subject str               record name, e.g. tr03-0005

Re-runnable: finished records are skipped. Use --limit N to process only N records.
"""

import argparse
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import (  # reuse the exact UCDDB pipeline steps
    TARGET_FS, WINDOW_SECONDS, lowpass_filter, resample_signal,
    segment_signal, zscore_segments,
)

DB = "challenge-2018/1.0.0/training"
MIRRORS = [
    f"https://physionet-open.s3.amazonaws.com/{DB}",   # fast AWS open-data mirror
    f"https://physionet.org/files/{DB}",               # fallback
]
RAW_DIR = Path("data/raw/cinc2018_tmp")
OUT_DIR = Path("data/processed/cinc2018")
CHANNELS = ["C4-M1", "C3-M2"]
APNEA_TYPES = {"resp_obstructiveapnea", "resp_centralapnea", "resp_mixedapnea", "resp_hypopnea"}
STAGES = {"W", "N1", "N2", "N3", "R"}
MIN_OVERLAP_S = 10.0


def curl(url_path, dest):
    for base in MIRRORS:
        r = subprocess.run(
            ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-sS",
             f"{base}/{url_path}", "-o", str(dest)],
            capture_output=True, text=True)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return
    raise RuntimeError(f"download failed: {url_path}")


def list_records():
    dest = RAW_DIR / "RECORDS"
    curl("RECORDS", dest)
    return [ln.strip().strip("/") for ln in dest.read_text().splitlines() if ln.strip()]


def parse_annotations(ann, fs, n_windows):
    """Returns (apnea events as (start_s, end_s, type), per-window stage, Counter of all labels)."""
    events, open_events, stage_marks, counts = [], {}, [], Counter()
    for sample, note in zip(ann.sample, ann.aux_note):
        note = note.strip()
        if not note:
            continue
        counts[note] += 1
        t = sample / fs
        if note in STAGES:
            stage_marks.append((t, note))
        elif note.startswith("("):
            open_events[note[1:]] = t
        elif note.endswith(")"):
            name = note[:-1]
            start = open_events.pop(name, None)
            if start is not None and name in APNEA_TYPES:
                events.append((start, t, name))

    stage = np.full(n_windows, "", dtype="<U2")
    stage_marks.sort()
    for i, (t, s) in enumerate(stage_marks):
        t_end = stage_marks[i + 1][0] if i + 1 < len(stage_marks) else n_windows * WINDOW_SECONDS
        a, b = int(round(t / WINDOW_SECONDS)), int(round(t_end / WINDOW_SECONDS))
        stage[max(a, 0):min(b, n_windows)] = s
    return events, stage, counts


def make_labels(n_windows, events):
    y = np.zeros(n_windows, dtype=np.int64)
    for start, end, _ in events:
        first = max(int(start // WINDOW_SECONDS), 0)
        last = min(int(end // WINDOW_SECONDS), n_windows - 1)
        for w in range(first, last + 1):
            ws, we = w * WINDOW_SECONDS, (w + 1) * WINDOW_SECONDS
            if min(we, end) - max(ws, start) >= MIN_OVERLAP_S:
                y[w] = 1
    return y


def window_channel(sig, fs):
    sig = np.nan_to_num(sig.astype(np.float64))
    sig = resample_signal(lowpass_filter(sig, fs), fs)
    return zscore_segments(segment_signal(sig))


def process_record(rec_dir, rec):
    """rec_dir holds rec.hea / rec.mat / rec.arousal. Returns a stats dict."""
    base = str(rec_dir / rec)
    header = wfdb.rdheader(base)
    missing = [c for c in CHANNELS if c not in header.sig_name]
    if missing:
        raise RuntimeError(f"{rec}: missing channels {missing} (has {header.sig_name})")
    record = wfdb.rdrecord(base, channel_names=CHANNELS, physical=True)
    fs = record.fs
    c4 = window_channel(record.p_signal[:, record.sig_name.index("C4-M1")], fs)
    c3 = window_channel(record.p_signal[:, record.sig_name.index("C3-M2")], fs)
    n = min(len(c4), len(c3))
    c4, c3 = c4[:n], c3[:n]

    ann = wfdb.rdann(base, "arousal")
    events, stage, counts = parse_annotations(ann, fs, n)
    y = make_labels(n, events)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / f"{rec}.tmp.npz"
    np.savez_compressed(tmp, X=c4, X_c3=c3.astype(np.float16), y=y, stage=stage, subject=rec)
    tmp.rename(OUT_DIR / f"{rec}.npz")
    return {"windows": n, "apnea_windows": int(y.sum()), "events": len(events), "counts": counts}


def download_and_process(rec):
    out = OUT_DIR / f"{rec}.npz"
    if out.exists():
        return rec, "skip", None
    rec_dir = RAW_DIR / rec
    rec_dir.mkdir(parents=True, exist_ok=True)
    try:
        for ext in (".hea", ".mat", ".arousal"):
            curl(f"{rec}/{rec}{ext}", rec_dir / f"{rec}{ext}")
        stats = process_record(rec_dir, rec)
        return rec, "done", stats
    finally:
        shutil.rmtree(rec_dir, ignore_errors=True)   # raw data never piles up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = list_records()
    if args.limit:
        records = records[:args.limit]
    todo = [r for r in records if not (OUT_DIR / f"{r}.npz").exists()]
    print(f"{len(records)} records listed, {len(records) - len(todo)} already processed, {len(todo)} to go")

    # Check the first new record on its own: the labels must contain apnea events.
    if todo:
        rec, status, stats = download_and_process(todo[0])
        resp = {k: v for k, v in stats["counts"].items() if "resp" in k}
        print(f"[check] {rec}: {stats['windows']} windows, {stats['events']} apnea/hypopnea events, "
              f"{stats['apnea_windows']} apnea windows")
        print(f"[check] respiratory labels in file: {dict(resp)}")
        if not resp:
            print("!! No respiratory-event labels found in the first record. Stopping so nothing "
                  "else is downloaded. All labels seen:", dict(stats["counts"]))
            sys.exit(2)
        todo = todo[1:]

    done = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_and_process, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = futures[fut]
            try:
                _, status, stats = fut.result()
                done += 1
                rate = (time.time() - t0) / i
                eta = rate * (len(todo) - i) / 60
                print(f"[{i}/{len(todo)}] {rec}: {stats['windows']} windows, "
                      f"{stats['apnea_windows']} apnea | ~{eta:.0f} min left", flush=True)
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(todo)}] !! {rec} FAILED: {e}", flush=True)

    shutil.rmtree(RAW_DIR, ignore_errors=True)
    files = list(OUT_DIR.glob("tr*.npz"))
    size_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"\nFinished: {len(files)} records in {OUT_DIR} ({size_gb:.1f} GB). "
          f"{failed} failed this run (re-run to retry them).")


if __name__ == "__main__":
    main()
