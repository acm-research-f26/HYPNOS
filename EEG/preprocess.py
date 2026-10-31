from pathlib import Path
import re

import numpy as np
import pyedflib

from scipy.signal import butter, sosfiltfilt, resample_poly


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

RAW_DIR = Path("data/raw/ucd")
OUTPUT_DIR = Path("data/processed")

SUBJECTS = [
    "005",
    "007",
    "012",
    "023",
    "028",
]

TARGET_FS = 125
WINDOW_SECONDS = 30
WINDOW_SAMPLES = TARGET_FS * WINDOW_SECONDS

LOWPASS_CUTOFF = 42.0


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def normalize_channel_name(name):
    """
    Makes channel-name matching easier.

    Example:
    C4-A1 -> C4A1
    C4 A1 -> C4A1
    """
    return "".join(
        char for char in name.upper()
        if char.isalnum()
    )


def time_to_seconds(time_string):
    """
    Converts HH:MM:SS into seconds from midnight.
    """

    hour, minute, second = map(
        int,
        time_string.split(":")
    )

    return (
        hour * 3600
        + minute * 60
        + second
    )


# --------------------------------------------------
# RESPIRATORY ANNOTATIONS
# --------------------------------------------------

def read_respiratory_events(annotation_path, recording_start):
    """
    Reads UCD respiratory-event annotations.

    Returns events as:

    {
        "start": seconds from beginning of recording,
        "duration": seconds,
        "type": event type
    }
    """

    events = []

    recording_start_seconds = (
        recording_start.hour * 3600
        + recording_start.minute * 60
        + recording_start.second
    )

    with open(
        annotation_path,
        "r",
        encoding="latin-1",
        errors="ignore"
    ) as file:

        for line in file:

            tokens = line.split()

            if len(tokens) < 3:
                continue

            time_token = tokens[0]

            if not re.fullmatch(
                r"\d{2}:\d{2}:\d{2}",
                time_token
            ):
                continue

            event_type = tokens[1].upper()

            # We only care about apnea / hypopnea events.
            if not (
                event_type.startswith("APNEA")
                or event_type.startswith("HYP")
            ):
                continue

            # Find the first number after event type.
            # This corresponds to event duration.
            duration = None

            for token in tokens[2:]:

                try:
                    possible_duration = float(token)

                    if possible_duration > 0:
                        duration = possible_duration
                        break

                except ValueError:
                    continue

            if duration is None:
                continue

            event_clock_seconds = time_to_seconds(
                time_token
            )

            relative_start = (
                event_clock_seconds
                - recording_start_seconds
            )

            # Most PSG recordings cross midnight.
            if relative_start < 0:
                relative_start += 24 * 3600

            events.append(
                {
                    "start": relative_start,
                    "duration": duration,
                    "type": event_type,
                }
            )

    return events


# --------------------------------------------------
# LABEL CREATION
# --------------------------------------------------

def create_labels(number_epochs, events):
    """
    Creates one label per 30-second EEG window.

    0 = non-apnea
    1 = apnea

    A segment is apnea if one respiratory event
    overlaps it continuously for at least 10 seconds.
    """

    labels = np.zeros(
        number_epochs,
        dtype=np.int64
    )

    for epoch_index in range(number_epochs):

        epoch_start = (
            epoch_index
            * WINDOW_SECONDS
        )

        epoch_end = (
            epoch_start
            + WINDOW_SECONDS
        )

        for event in events:

            event_start = event["start"]

            event_end = (
                event["start"]
                + event["duration"]
            )

            overlap = max(
                0.0,
                min(epoch_end, event_end)
                - max(epoch_start, event_start)
            )

            if overlap >= 10.0:
                labels[epoch_index] = 1
                break

    return labels


# --------------------------------------------------
# SIGNAL PREPROCESSING
# --------------------------------------------------

def lowpass_filter(signal, sampling_rate):
    """
    Low-pass EEG before resampling.
    """

    cutoff = min(
        LOWPASS_CUTOFF,
        sampling_rate / 2 - 1
    )

    sos = butter(
        N=4,
        Wn=cutoff,
        btype="lowpass",
        fs=sampling_rate,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


def resample_signal(signal, original_fs):
    """
    Resamples EEG to 125 Hz.
    """

    original_fs = int(
        round(original_fs)
    )

    if original_fs == TARGET_FS:
        return signal

    return resample_poly(
        signal,
        TARGET_FS,
        original_fs
    )


def segment_signal(signal):
    """
    Converts continuous EEG into 30-second segments.
    """

    number_epochs = (
        len(signal)
        // WINDOW_SAMPLES
    )

    usable_length = (
        number_epochs
        * WINDOW_SAMPLES
    )

    signal = signal[:usable_length]

    segments = signal.reshape(
        number_epochs,
        WINDOW_SAMPLES
    )

    return segments


def zscore_segments(segments):
    """
    Standardizes every 30-second EEG segment.
    """

    means = np.mean(
        segments,
        axis=1,
        keepdims=True
    )

    standard_deviations = np.std(
        segments,
        axis=1,
        keepdims=True
    )

    standard_deviations[
        standard_deviations < 1e-8
    ] = 1.0

    normalized = (
        segments - means
    ) / standard_deviations

    return normalized.astype(
        np.float32
    )


# --------------------------------------------------
# PROCESS ONE SUBJECT
# --------------------------------------------------

def process_subject(subject):
    print()
    print("=" * 60)
    print(f"Processing UCD subject {subject}")
    print("=" * 60)

    eeg_path = (
        RAW_DIR
        / f"ucddb{subject}.rec"
    )

    annotation_path = (
        RAW_DIR
        / f"ucddb{subject}_respevt.txt"
    )

    if not eeg_path.exists():
        raise FileNotFoundError(
            eeg_path
        )

    if not annotation_path.exists():
        raise FileNotFoundError(
            annotation_path
        )

    # ----------------------------
    # OPEN EDF
    # ----------------------------

    reader = pyedflib.EdfReader(
        str(eeg_path)
    )

    channel_names = (
        reader.getSignalLabels()
    )

    print("Available channels:")

    for index, channel in enumerate(
        channel_names
    ):
        print(index, channel)

    # ----------------------------
    # FIND C4-A1
    # ----------------------------

    c4_index = None

    for index, channel in enumerate(
        channel_names
    ):

        normalized_name = (
            normalize_channel_name(
                channel
            )
        )

        if normalized_name == "C4A1":
            c4_index = index
            break

    if c4_index is None:

        reader.close()

        raise RuntimeError(
            "Could not find C4-A1 EEG channel."
        )

    print()
    print(
        "Using channel:",
        channel_names[c4_index]
    )

    sampling_rate = (
        reader.getSampleFrequency(
            c4_index
        )
    )

    print(
        "Original sampling rate:",
        sampling_rate
    )

    recording_start = (
        reader.getStartdatetime()
    )

    print(
        "Recording start:",
        recording_start
    )

    eeg = reader.readSignal(
        c4_index
    ).astype(
        np.float64
    )

    reader.close()

    print(
        "Raw samples:",
        len(eeg)
    )

    # ----------------------------
    # ANNOTATIONS
    # ----------------------------

    events = (
        read_respiratory_events(
            annotation_path,
            recording_start
        )
    )

    print(
        "Respiratory events:",
        len(events)
    )

    # ----------------------------
    # FILTER
    # ----------------------------

    print(
        "Applying low-pass filter..."
    )

    eeg = lowpass_filter(
        eeg,
        sampling_rate
    )

    # ----------------------------
    # RESAMPLE
    # ----------------------------

    print(
        f"Resampling to {TARGET_FS} Hz..."
    )

    eeg = resample_signal(
        eeg,
        sampling_rate
    )

    # ----------------------------
    # SEGMENT
    # ----------------------------

    segments = segment_signal(
        eeg
    )

    print(
        "EEG segments:",
        segments.shape
    )

    # ----------------------------
    # NORMALIZE
    # ----------------------------

    segments = zscore_segments(
        segments
    )

    # ----------------------------
    # LABELS
    # ----------------------------

    labels = create_labels(
        len(segments),
        events
    )

    apnea_count = int(
        np.sum(labels == 1)
    )

    non_apnea_count = int(
        np.sum(labels == 0)
    )

    print(
        "Apnea segments:",
        apnea_count
    )

    print(
        "Non-apnea segments:",
        non_apnea_count
    )

    # ----------------------------
    # SAVE
    # ----------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"ucddb{subject}.npz"
    )

    np.savez_compressed(
        output_path,
        X=segments,
        y=labels,
        subject=subject
    )

    print(
        "Saved:",
        output_path
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for subject in SUBJECTS:

        try:
            process_subject(
                subject
            )

        except Exception as error:

            print()
            print(
                f"ERROR processing subject {subject}:"
            )

            print(error)

            raise

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()