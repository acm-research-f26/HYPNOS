"""Tests for device_pipeline.py: python -m unittest test_device_pipeline"""
import unittest

import numpy as np
import torch

import apnea
import device_pipeline as D


def tone(freq, seconds, amp=20.0, fs=D.FS_IN):
    return amp * np.sin(2 * np.pi * freq * np.arange(int(seconds * fs)) / fs)[:, None]


def amplitude(freq, mains=60.0):
    """Output std (microvolts) of a unit-amplitude 20 uV tone after the pipeline."""
    wins = D.Preprocessor(1, mains=mains, warmup_s=5).push(tone(freq, 125))
    return float(np.mean([w["std_uv"][0] for w in wins])) / (20 / np.sqrt(2))


class PreprocessorTests(unittest.TestCase):
    def test_chunking_does_not_change_the_output(self):
        x = np.random.default_rng(0).normal(0, 10, (250 * 100, 3))
        whole = D.Preprocessor(3).push(x)
        pre, got, i = D.Preprocessor(3), [], 0
        for n in np.random.default_rng(1).integers(1, 777, 10000):
            if i >= len(x):
                break
            got += pre.push(x[i:i + n])
            i += n
        self.assertEqual(len(whole), len(got))
        for a, b in zip(whole, got):
            np.testing.assert_allclose(a["x"], b["x"], atol=1e-5)

    def test_window_count_and_shape(self):
        wins = D.Preprocessor(4, warmup_s=0).push(np.random.default_rng(0).normal(size=(250 * 95, 4)))
        self.assertEqual(len(wins), 3)  # 95 s -> three whole 30 s windows
        self.assertEqual(wins[0]["x"].shape, (4, D.WIN))

    def test_passband_and_stopband(self):
        self.assertGreater(amplitude(10), 0.85)  # alpha passes
        self.assertLess(amplitude(1), 0.02)  # drift removed
        self.assertLess(amplitude(60), 0.05)  # 60 Hz mains notched
        self.assertGreater(amplitude(60, mains=0), 0.05)  # ...and only because of the notch

    def test_windows_are_zscored_per_channel(self):
        w = D.Preprocessor(2, warmup_s=0).push(np.random.default_rng(0).normal(0, 10, (250 * 30, 2)))[0]
        np.testing.assert_allclose(w["x"].mean(1), 0, atol=1e-4)
        np.testing.assert_allclose(w["x"].std(1), 1, atol=1e-3)

    def test_flat_stuck_and_nonfinite_channels_are_flagged(self):
        x = np.random.default_rng(0).normal(0, 10, (250 * 30, 4))
        x[:, 1] = 0.0  # disconnected: flat
        x[:, 2] = np.where(np.arange(len(x)) % 2 == 0, 5.0, 5.0)  # stuck
        x[100, 3] = np.nan
        w = D.Preprocessor(4, warmup_s=0).push(x)[0]
        self.assertEqual(w["ok"].tolist(), [True, False, False, False])
        self.assertFalse(w["x"][1:].any())  # unusable channels are zeroed


def write_xdf(path, x, ts, name="Cerelog_EEG", fs=D.FS_IN):
    """Minimal one-stream XDF file (float32 samples) in the layout LabRecorder writes."""
    import struct

    def chunk(tag, body):
        n = len(body) + 2
        return (b"\x01" + bytes([n]) if n < 256 else b"\x04" + struct.pack("<I", n)) + struct.pack("<H", tag) + body
    hdr = (f"<?xml version='1.0'?><info><name>{name}</name><type>EEG</type><channel_count>{x.shape[1]}</channel_count>"
           f"<nominal_srate>{fs}</nominal_srate><channel_format>float32</channel_format></info>").encode()
    out = b"XDF:" + chunk(1, b"<?xml version='1.0'?><info><version>1.0</version></info>") + chunk(2, struct.pack("<I", 1) + hdr)
    body = struct.pack("<IB", 1, 4) + struct.pack("<I", len(x))
    body += b"".join(b"\x08" + struct.pack("<d", t) + r.astype(np.float32).tobytes() for r, t in zip(x, ts))
    with open(path, "wb") as f:
        f.write(out + chunk(3, body))


class XdfTests(unittest.TestCase):
    def test_xdf_matches_the_csv_log_of_the_same_stream(self):
        """An .xdf holds the stream values, the logger CSV holds ADC counts; both must give the same microvolts."""
        import tempfile, os
        uv = np.random.default_rng(0).normal(0, 20, (250 * 40, 8))
        stream = (uv * D.GUI_CORRECTION_FACTOR).astype(np.float32)  # what cerelog_lsl.py publishes
        counts = np.round(stream / (D.UV_PER_COUNT * D.GUI_CORRECTION_FACTOR))  # what lsl_to_csv_logmaker.py writes
        ts = 100 + np.arange(len(uv)) / D.FS_IN
        with tempfile.TemporaryDirectory() as d:
            xp, cp, q = os.path.join(d, "a.xdf"), os.path.join(d, "b.xdf"), os.path.join(d, "n.csv")
            write_xdf(xp, stream, ts)
            np.savetxt(cp, np.column_stack([np.round((ts - ts[0]) * 1000), counts]), delimiter=",", fmt="%d",
                       header="Timestamp," + ",".join(f"CH{i + 1}" for i in range(8)), comments="")
            x, t = D.read_xdf(xp, [2, 5])
            c = D.read_csv(cp, [2, 5])
        np.testing.assert_allclose(x, uv[:, [2, 5]], rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(c, uv[:, [2, 5]], atol=0.05)  # one ADC count is ~0.02 uV
        np.testing.assert_allclose(t, ts)
        a, b = D.Preprocessor(2).push(x)[0], D.Preprocessor(2).push(c)[0]
        np.testing.assert_allclose(a["x"], b["x"], atol=5e-3)

    def test_stream_fallback_and_channel_check(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            p, q = os.path.join(d, "a.xdf"), os.path.join(d, "b.xdf")
            write_xdf(p, np.zeros((10, 8)), np.arange(10.0), name="Other")
            write_xdf(q, np.zeros((10, 8)), np.arange(10.0), name="Cerelog_EEG")
            # a single EEG stream under another name is still used
            self.assertEqual(D.read_xdf(p, [0])[0].shape, (10, 1))
            with self.assertRaises(SystemExit):
                D.read_xdf(q, [8])  # channel out of range


class ScorerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = apnea.MC(K=1).to(apnea.DEV).eval()

    def windows(self, n, ok=(True, True)):
        rng = np.random.default_rng(0)
        return [dict(index=i, x=rng.normal(size=(2, D.WIN)).astype(np.float32), ok=np.array(ok), std_uv=np.ones(2))
                for i in range(n)]

    def test_every_window_is_scored_once_in_order(self):
        s, out = D.Scorer(self.model), []
        for w in self.windows(5):
            out += s.push(w)
        out += s.flush()
        self.assertEqual([r["index"] for r in out], [0, 1, 2, 3, 4])
        self.assertTrue(all(np.isfinite(r["logit"]) for r in out))

    def test_matches_the_model_on_prev_current_next(self):
        ws = self.windows(3)
        s, out = D.Scorer(self.model), []
        for w in ws:
            out += s.push(w)
        x = torch.from_numpy(np.stack([w["x"] for w in ws], 1)[None]).to(apnea.DEV)  # [1, C, 3, T]
        with torch.no_grad():
            want = self.model(x, torch.ones(1, 2, device=apnea.DEV))[0].item()
        self.assertEqual([r["index"] for r in out], [0, 1])  # window 1 has both neighbours; window 2 waits for flush()
        self.assertAlmostEqual(out[1]["logit"], want, places=4)

    def test_unusable_centre_is_reported_not_scored(self):
        ws = self.windows(3)
        ws[1]["ok"] = np.array([False, False])
        s, out = D.Scorer(self.model), []
        for w in ws:
            out += s.push(w)
        out += s.flush()
        self.assertEqual([r["usable"] for r in out], [True, False, True])


if __name__ == "__main__":
    unittest.main()
