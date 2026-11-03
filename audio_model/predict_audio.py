
import argparse
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly

from osa_audio_model_30s import process_segment


# HYPNOS settings
SAMPLE_RATE = 16000
WINDOW_SEC = 30
STRIDE_SEC = 10

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "hypnos_audio_v2.keras"
THRESHOLD_PATH = BASE_DIR / "best_threshold.txt"
OUTPUT_PATH = BASE_DIR / "audio_predictions.csv"


def load_model_and_threshold():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing threshold: {THRESHOLD_PATH}"
        )

    model = tf.keras.models.load_model(str(MODEL_PATH))

    threshold = float(
        THRESHOLD_PATH.read_text().strip()
    )

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    print("Model loaded successfully.")
    print("Classification threshold:", threshold)

    return model, threshold


def process_audio_window(segment, original_rate):
    # Stereo to mono
    if segment.ndim == 2:
        segment = segment.mean(axis=1)

    # Resample audio
    if original_rate != SAMPLE_RATE:
        divisor = gcd(original_rate, SAMPLE_RATE)

        segment = resample_poly(
            segment,
            SAMPLE_RATE // divisor,
            original_rate // divisor
        )

    expected_samples = WINDOW_SEC * SAMPLE_RATE

    if len(segment) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, "
            f"received {len(segment)}"
        )

    # Reuse training preprocessing
    features = process_segment(segment)

    features = np.asarray(
        features, dtype=np.float32
    )

    if features.ndim != 2:
        raise ValueError(
            f"Expected a 2D spectrogram, got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError("Invalid spectrogram values")

    # Add CNN channel and batch dimensions
    features = features[np.newaxis, ..., np.newaxis]

    return features


def predict_recording(audio_path):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model, threshold = load_model_and_threshold()

    predictions = []

    with sf.SoundFile(str(audio_path)) as audio:
        original_rate = audio.samplerate
        duration = len(audio) / original_rate

        if duration < WINDOW_SEC:
            raise ValueError(
                "Recording must be at least 30 seconds long"
            )

        print("Audio file:", audio_path)
        print("Original sample rate:", original_rate)
        print("Duration:", round(duration, 2), "seconds")

        # Read only one window at a time
        for start_sec in range(
            0,
            int(duration - WINDOW_SEC) + 1,
            STRIDE_SEC
        ):
            end_sec = start_sec + WINDOW_SEC

            audio.seek(int(start_sec * original_rate))

            segment = audio.read(
                WINDOW_SEC * original_rate,
                dtype="float32"
            )

            features = process_audio_window(
                segment, original_rate
            )

            probability = float(
                model.predict(features, verbose=0)[0][0]
            )

            prediction = int(probability >= threshold)

            predictions.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "probability": probability,
                "prediction": prediction
            })

    results = pd.DataFrame(predictions)

    results.to_csv(OUTPUT_PATH, index=False)

    print("\nHYPNOS AUDIO ANALYSIS COMPLETED")
    print("Total windows:", len(results))
    print(
        "Positive windows:",
        int(results["prediction"].sum())
    )
    print(
        "Negative windows:",
        int((results["prediction"] == 0).sum())
    )
    print("Saved predictions:", OUTPUT_PATH)

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "audio_file",
        help="Path to a WAV audio recording"
    )

    args = parser.parse_args()

    predict_recording(args.audio_file)


if __name__ == "__main__":
    main()
