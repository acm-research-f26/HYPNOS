"""Device-side preprocessing and live scoring for the Cerelog ADS1299 board (250 Hz, wet electrodes).

Raw samples [n, C] in microvolts -> causal 5-42 Hz band-pass + mains notch -> 125 Hz -> 30 s windows, z-scored per
channel and quality-checked -> optional rolling model scores. It mirrors the training data (RESULTS.md): the filter must
match what the model was trained on, since a causal/zero-phase mismatch cost 0.15-0.20 AUROC.

Reference only (not imported, not modified): EEG/cerelog_lsl.py and EEG/lsl_to_csv_logmaker.py on the eeg-firmware branch.
Facts taken from them: the LSL stream 'Cerelog_EEG' has 8 channels at 250 Hz, values are microvolts times
GUI_CORRECTION_FACTOR, a ~0.2 Hz DC blocker is already applied, and sample times are nominal (dropped packets are not
visible as gaps).

    python device_pipeline.py --csv log.csv --channels 0,1,2,3 --model mc_hp5_big --mains 60
    python device_pipeline.py --xdf night.xdf --channels 0,1,2,3 --model mc_hp5_big    (LabRecorder file, needs pyxdf)
    python device_pipeline.py --lsl --channels 0,1,2,3 --model mc_hp5_big
"""
import argparse
from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

FS_IN, FS_OUT, WIN = 250, 125, 3750  # board rate, model rate, samples per 30 s window at FS_OUT
# Constants from cerelog_lsl.py (eeg-firmware branch); keep in sync with the firmware script
HARDWARE_VREF, HARDWARE_GAIN = 4.50, 24
GUI_CORRECTION_FACTOR = 0.94 * 24.0 / 10 / 2  # the LSL stream is scaled by this on top of true microvolts
UV_PER_COUNT = (2 * HARDWARE_VREF / HARDWARE_GAIN) / 2 ** 24 * 1e6  # logger CSV holds ADC counts
FLAT_UV = 0.5  # filtered std below this (microvolts) = flat channel
STUCK_FRAC = 0.5  # more than this share of identical consecutive samples = stuck / clipped (same rule as build_cache.py)


def design(hp=5.0, lp=42.0, mains=60.0, order=4):
    """Causal band-pass (Butterworth, `order` poles per edge) followed by an optional mains notch, as SOS sections."""
    sos = butter(order, [hp, lp], "bandpass", fs=FS_IN, output="sos")
    if mains:
        sos = np.vstack([sos, tf2sos(*iirnotch(mains, 30.0, fs=FS_IN))])
    return sos


def artifact_ok(raw_uv):
    """Return False if this filtered channel-window [T] (microvolts) holds an artifact you want to reject.

    TODO(human): decide the artifact policy for your wet electrodes (peak amplitude? a robust z-score? muscle-band
    power?). Training only removed flat, stuck and non-finite windows, so the default (accept everything) matches it.
    """
    return True


def window_ok(raw_uv):
    """Same cleaning as the training data (build_cache.py), plus the artifact policy above."""
    return bool(np.isfinite(raw_uv).all() and raw_uv.std() >= FLAT_UV
                and (np.diff(raw_uv) == 0).mean() <= STUCK_FRAC and artifact_ok(raw_uv))


class Preprocessor:
    """Stateful: feed chunks of any size to push(); each completed 30 s window comes back as a dict."""

    def __init__(self, n_ch, hp=5.0, lp=42.0, mains=60.0, warmup_s=5.0):
        self.sos = design(hp, lp, mains)
        self.zi = np.zeros((len(self.sos), 2, n_ch))  # filter state carried across chunks
        self.skip = int(warmup_s * FS_IN)  # samples still to drop while the filter settles
        self.phase = 0  # decimation phase carried across chunks
        self.buf = np.empty((0, n_ch))
        self.n_win = 0

    def push(self, x):
        """x [n, C] in true microvolts at 250 Hz -> list of finished windows."""
        y, self.zi = sosfilt(self.sos, np.asarray(x, float), axis=0, zi=self.zi)
        drop = min(self.skip, len(y))
        y, self.skip = y[drop:], self.skip - drop
        y, self.phase = y[self.phase::2], (self.phase + len(y)) % 2  # 250 -> 125 Hz; the 42 Hz low-pass is the anti-alias
        self.buf = np.vstack([self.buf, y])
        out = []
        while len(self.buf) >= WIN:
            out.append(self._finish(self.buf[:WIN].T))
            self.buf = self.buf[WIN:]
        return out

    def _finish(self, raw):  # raw [C, T] filtered microvolts
        ok = np.array([window_ok(c) for c in raw])
        z = (raw - raw.mean(1, keepdims=True)) / (raw.std(1, keepdims=True) + 1e-6)
        z[~ok] = 0
        self.n_win += 1
        return dict(index=self.n_win - 1, x=z.astype(np.float32), ok=ok, std_uv=raw.std(1))


class Scorer:
    """Rolling scores with the model's prev/current/next context. Needs the next window, so each window is scored
    K windows (30 s each) late; a missing or unusable neighbour is replaced by the centre window, as in training."""

    def __init__(self, model):
        import apnea
        self.apnea, self.model, self.K = apnea, model.eval(), model.K
        self.hist = deque([(None, None)] * self.K, maxlen=2 * self.K + 1)

    def _embed(self, win):
        import torch
        ok = np.flatnonzero(win["ok"])
        if not len(ok):
            return None
        if not isinstance(self.model, self.apnea.MC):  # single-channel models use the first usable channel
            ok = ok[:1]
        with torch.no_grad():
            return self.model.encode(torch.from_numpy(win["x"][ok][None]).to(self.apnea.DEV))[0]

    def push(self, win):
        self.hist.append((self._embed(win), win) if win is not None else (None, None))
        return self._score() if len(self.hist) == 2 * self.K + 1 else []

    def flush(self):
        out = []
        for _ in range(self.K):
            out += self.push(None)
        return out

    def _score(self):
        import torch
        centre, win = self.hist[self.K]
        if win is None:
            return []
        if centre is None:
            return [dict(index=win["index"], usable=False, logit=float("nan"), n_ch=0)]
        es = torch.stack([e if e is not None else centre for e, _ in self.hist])
        with torch.no_grad():
            logit = self.model.combine(es[None])[0][0].item()
        return [dict(index=win["index"], usable=True, logit=logit, n_ch=int(win["ok"].sum()))]


def read_csv(path, channels):
    """Logger CSV (Timestamp ms, CH1..CHn in ADC counts) -> true microvolts [n, len(channels)]."""
    return np.loadtxt(path, delimiter=",", skiprows=1)[:, 1:][:, channels] * UV_PER_COUNT


def read_xdf(path, channels, name="Cerelog_EEG"):
    """LabRecorder .xdf of the LSL stream -> (true microvolts [n, len(channels)], timestamps in s).
    Uses the stream called `name`, else the only EEG-type stream. Values are what the stream carried, so they are
    divided by GUI_CORRECTION_FACTOR like stream_lsl() does."""
    import pyxdf
    streams, _ = pyxdf.load_xdf(str(path))
    eeg = [s for s in streams if s["info"]["type"][0].lower() == "eeg"]
    pick = [s for s in eeg if s["info"]["name"][0] == name] or (eeg if len(eeg) == 1 else [])
    if not pick:
        found = ", ".join(f"{s['info']['name'][0]} ({s['info']['type'][0]})" for s in streams) or "none"
        raise SystemExit(f"No '{name}' stream in {path} (streams: {found})")
    s = pick[0]
    fs = float(s["info"]["nominal_srate"][0])
    if fs != FS_IN:
        raise SystemExit(f"Stream is {fs:g} Hz; this pipeline expects the board's {FS_IN} Hz")
    x = np.asarray(s["time_series"], float)
    if max(channels) >= x.shape[1]:
        raise SystemExit(f"Channel {max(channels)} out of range: the stream has channels 0..{x.shape[1] - 1}")
    return x[:, channels] / GUI_CORRECTION_FACTOR, np.asarray(s["time_stamps"], float)


def check_timing(t):
    """Print the rate implied by the timestamps and any gaps. cerelog_lsl.py stamps samples on a fixed schedule,
    so dropped packets usually do NOT show up here; a clean result does not prove nothing was lost."""
    if len(t) < 2:
        return
    span, d = t[-1] - t[0], np.diff(t)
    print(f"{len(t)} samples over {span / 3600:.2f} h, timestamps imply {(len(t) - 1) / span:.2f} Hz (nominal {FS_IN})")
    gaps = np.flatnonzero(d > 2.0 / FS_IN)
    if len(gaps):
        print(f"  warning: {len(gaps)} gap(s), {d[gaps].sum() - len(gaps) / FS_IN:.1f} s missing in total; "
              f"windows spanning a gap join two stretches of signal")


def stream_lsl(channels, name="Cerelog_EEG", chunk=250):
    """Yield chunks [n, len(channels)] in true microvolts from the live LSL stream (needs pylsl)."""
    from pylsl import StreamInlet, resolve_byprop
    inlet = StreamInlet(resolve_byprop("name", name, timeout=30)[0], max_buflen=60)
    while True:
        samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=chunk)
        if samples:
            yield np.asarray(samples)[:, channels] / GUI_CORRECTION_FACTOR


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--xdf", help="LabRecorder recording of the Cerelog_EEG stream")
    src.add_argument("--lsl", action="store_true")
    p.add_argument("--channels", default="0,1,2,3", help="board channels to use, 0-based")
    p.add_argument("--model", help="checkpoint name for apnea.load (omit to only preprocess)")
    p.add_argument("--mains", type=float, default=60.0, help="mains frequency in Hz (0 = no notch)")
    p.add_argument("--thr", type=float, help="logit threshold; windows at or above it are flagged")
    p.add_argument("--out", help="CSV for the per-window scores")
    a = p.parse_args()
    chans = [int(c) for c in a.channels.split(",")]
    pre = Preprocessor(len(chans), mains=a.mains)
    scorer = None
    if a.model:
        import apnea
        scorer = Scorer(apnea.load(a.model))
    rows = []

    def handle(results):
        for r in results:
            flag = "" if a.thr is None or not r["usable"] else ("  APNEA?" if r["logit"] >= a.thr else "")
            print(f"window {r['index']:4d} ({r['index'] / 2:6.1f} min)  channels {r['n_ch']}  logit {r['logit']:+.3f}{flag}", flush=True)
            rows.append(r)

    if a.xdf:
        x, t = read_xdf(a.xdf, chans)
        check_timing(t)
        chunks = iter([x])
    else:
        chunks = iter([read_csv(a.csv, chans)]) if a.csv else stream_lsl(chans)
    try:
        for x in chunks:
            for win in pre.push(x):
                if scorer:
                    handle(scorer.push(win))
                else:
                    print(f"window {win['index']:4d}  usable {win['ok'].astype(int)}  std {np.round(win['std_uv'], 1)} uV")
    except KeyboardInterrupt:
        pass
    if scorer:
        handle(scorer.flush())
    if a.out and rows:
        np.savetxt(a.out, [[r["index"], r["n_ch"], r["logit"]] for r in rows], delimiter=",", header="window,channels,logit", comments="")


if __name__ == "__main__":
    main()
