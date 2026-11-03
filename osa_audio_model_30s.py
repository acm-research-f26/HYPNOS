import librosa
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)


# ============================================================
# HYPNOS - OSA AUDIO MODEL
#
# Reproduction based on:
# "Acoustic Screening for Obstructive Sleep Apnea
# in Home Environments Based on Deep Neural Networks"
#
# This is NOT the authors' original source code.
# ============================================================


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

SAMPLE_RATE = 16000

SEGMENT_SECONDS = 30
SEGMENT_SHIFT_SECONDS = 10

N_MELS = 64

FMIN = 70
FMAX = 7500

WIN_LENGTH = 800
HOP_LENGTH = 320
N_FFT = 800


# ------------------------------------------------------------
# 1. LOAD AUDIO
# ------------------------------------------------------------

def load_audio(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return audio, sr


# ------------------------------------------------------------
# 2. CREATE MEL SPECTROGRAM
# ------------------------------------------------------------

def make_mel_spectrogram(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    return mel


# ------------------------------------------------------------
# 3. CONVERT TO DECIBELS
# ------------------------------------------------------------

def convert_to_db(mel):
    return librosa.power_to_db(
        mel,
        ref=np.max
    )


# ------------------------------------------------------------
# 4. NORMALIZE FEATURES
# ------------------------------------------------------------

def normalize_features(mel_db):
    mean = np.mean(mel_db)
    std = np.std(mel_db)

    if std == 0:
        return mel_db

    return (mel_db - mean) / std


# ------------------------------------------------------------
# 5. SPLIT AUDIO INTO 30-SECOND SEGMENTS
# ------------------------------------------------------------

def split_audio_into_segments(audio):
    segment_samples = SAMPLE_RATE * SEGMENT_SECONDS
    shift_samples = SAMPLE_RATE * SEGMENT_SHIFT_SECONDS

    segments = []

    start = 0

    while start + segment_samples <= len(audio):
        end = start + segment_samples

        segment = audio[start:end]

        segments.append(segment)

        start += shift_samples

    return segments


# ------------------------------------------------------------
# 6. PROCESS ONE SEGMENT
# ------------------------------------------------------------

def process_segment(segment):
    mel = make_mel_spectrogram(segment)

    mel_db = convert_to_db(mel)

    normalized = normalize_features(mel_db)

    return normalized


# ------------------------------------------------------------
# 7. PREPARE AUDIO FOR CNN
# ------------------------------------------------------------

def prepare_audio_for_model(audio):
    segments = split_audio_into_segments(audio)

    processed_segments = []

    for segment in segments:
        processed = process_segment(segment)
        processed_segments.append(processed)

    if len(processed_segments) == 0:
        return np.array([])

    X = np.array(processed_segments)

    # Add CNN channel dimension
    X = X[..., np.newaxis]

    return X


# ------------------------------------------------------------
# 8. TEST PREPROCESSING
# ------------------------------------------------------------

def test_audio_processing(file_path):
    print()
    print("------------------------------")
    print("TESTING AUDIO PROCESSING")
    print("------------------------------")

    audio, sr = load_audio(file_path)

    print("Loaded audio successfully")
    print("Number of audio samples:", len(audio))
    print("Actual sample rate:", sr)

    mel = make_mel_spectrogram(audio)

    print("Mel spectrogram shape:", mel.shape)

    mel_db = convert_to_db(mel)

    print("Mel dB shape:", mel_db.shape)

    mel_normalized = normalize_features(mel_db)

    print(
        "Normalized Mel shape:",
        mel_normalized.shape
    )

    print(
        "Normalized mean:",
        np.mean(mel_normalized)
    )

    print(
        "Normalized standard deviation:",
        np.std(mel_normalized)
    )

    segments = split_audio_into_segments(audio)

    print()
    print(
        "Number of 30-second segments:",
        len(segments)
    )

    processed_segments = []

    for i, segment in enumerate(segments):
        processed = process_segment(segment)

        processed_segments.append(processed)

        print(
            "Processed segment",
            i + 1,
            "shape:",
            processed.shape
        )

    return audio, segments, processed_segments


# ------------------------------------------------------------
# 9. BUILD CNN MODEL
# ------------------------------------------------------------

def build_osa_model(input_shape):
    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(
        Conv2D(
            16,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 3)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 3)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 3)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(
        Dense(
            512,
            activation="relu"
        )
    )

    model.add(Dropout(0.3))

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------------------------------------------
# 10. TRAIN MODEL
# ------------------------------------------------------------

def train_model(model, X_train, y_train):
    history = model.fit(
        X_train,
        y_train,
        batch_size=128,
        epochs=50,
        validation_split=0.2,
        shuffle=True
    )

    return history


# ------------------------------------------------------------
# 11. PREDICT SEGMENTS
# ------------------------------------------------------------

def predict_segments(model, processed_segments):
    if len(processed_segments) == 0:
        return np.array([]), np.array([])

    probabilities = model.predict(
        processed_segments,
        verbose=0
    )

    probabilities = probabilities.flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return probabilities, predictions


# ------------------------------------------------------------
# 12. COUNT PREDICTED EVENTS
# ------------------------------------------------------------

def count_predicted_events(predictions):
    if len(predictions) == 0:
        return 0

    event_count = 0
    inside_event = False

    for prediction in predictions:

        if prediction == 1 and not inside_event:
            event_count += 1
            inside_event = True

        elif prediction == 0:
            inside_event = False

    return event_count


# ------------------------------------------------------------
# 13. ESTIMATE AHI
# ------------------------------------------------------------

def estimate_ahi(event_count, audio_duration_seconds):
    hours = audio_duration_seconds / 3600

    if hours <= 0:
        return 0

    return event_count / hours


# ------------------------------------------------------------
# 14. CLASSIFY OSA SEVERITY
# ------------------------------------------------------------

def classify_osa_severity(ahi):
    if ahi < 5:
        return "Normal / No OSA"

    elif ahi < 15:
        return "Mild OSA"

    elif ahi < 30:
        return "Moderate OSA"

    else:
        return "Severe OSA"


# ------------------------------------------------------------
# 15. ANALYZE A FULL SLEEP AUDIO FILE
# ------------------------------------------------------------

def analyze_sleep_audio(file_path, model):
    print()
    print("==============================")
    print("ANALYZING SLEEP AUDIO")
    print("==============================")

    audio, sr = load_audio(file_path)

    duration = len(audio) / sr

    print(
        "Audio duration:",
        round(duration, 2),
        "seconds"
    )

    X = prepare_audio_for_model(audio)

    print(
        "Number of segments:",
        len(X)
    )

    if len(X) == 0:
        print(
            "Audio is too short for a",
            SEGMENT_SECONDS,
            "second segment."
        )

        return None

    probabilities, predictions = predict_segments(
        model,
        X
    )

    for i in range(len(predictions)):
        print(
            "Segment",
            i + 1,
            "| Probability:",
            round(float(probabilities[i]), 3),
            "| Prediction:",
            int(predictions[i])
        )

    event_count = count_predicted_events(
        predictions
    )

    ahi = estimate_ahi(
        event_count,
        duration
    )

    severity = classify_osa_severity(
        ahi
    )

    print()
    print("------------------------------")
    print("RESULT")
    print("------------------------------")

    print(
        "Predicted events:",
        event_count
    )

    print(
        "Estimated AHI:",
        round(ahi, 2)
    )

    print(
        "OSA classification:",
        severity
    )

    return {
        "event_count": event_count,
        "ahi": ahi,
        "severity": severity,
        "probabilities": probabilities,
        "predictions": predictions
    }


# ------------------------------------------------------------
# 16. SAVE MODEL
# ------------------------------------------------------------

def save_model(model, file_name="osa_model.keras"):
    model.save(file_name)

    print(
        "Model saved as:",
        file_name
    )


# ------------------------------------------------------------
# 17. LOAD SAVED MODEL
# ------------------------------------------------------------

def load_saved_model(file_name="osa_model.keras"):
    model = tf.keras.models.load_model(
        file_name
    )

    return model


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print("HYPNOS OSA audio model started")
    print("Sample rate:", SAMPLE_RATE)
    print("Segment length:", SEGMENT_SECONDS, "seconds")
    print(
        "Segment shift:",
        SEGMENT_SHIFT_SECONDS,
        "seconds"
    )
    print("Mel filters:", N_MELS)

    audio, segments, processed_segments = test_audio_processing(
        "test_audio.wav"
    )

    print()
    print("------------------------------")
    print("BUILDING CNN")
    print("------------------------------")

    model = build_osa_model(
        (64, 1501, 1)
    )

    model.summary()

    print()
    print(
        "CNN successfully built."
    )

    print()
    print(
        "IMPORTANT: The model has NOT been trained yet."
    )

    print(
        "Do not interpret its predictions as real OSA results."
    )