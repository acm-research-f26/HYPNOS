
from pathlib import Path
import pandas as pd
import soundfile as sf

DATASET = Path.home() / "Downloads" / "APSAA"
OUTPUT = Path("training_windows.csv")

WINDOW_SEC = 30
STRIDE_SEC = 10

EVENT_TYPES = {
    "Obstructive Apnea",
    "Hypopnea"
}

rows = []

for folder in sorted(DATASET.iterdir()):
    if not folder.is_dir():
        continue

    subject_id = folder.name
    wav = folder / f"{subject_id}.wav"
    csv = folder / f"{subject_id}_Annotations.csv"

    if not wav.exists() or not csv.exists():
        print("Skipping incomplete folder:", subject_id)
        continue

    info = sf.info(str(wav))
    annotations = pd.read_csv(csv)

    annotations["start_sec"] = pd.to_timedelta(
        annotations["Start_Time"]
    ).dt.total_seconds()

    annotations["end_sec"] = (
        annotations["start_sec"] +
        annotations["Duration"]
    )

    events = annotations[
        annotations["Event_Name"].isin(EVENT_TYPES)
    ]

    positive = 0
    negative = 0

    for start in range(0, int(info.duration - WINDOW_SEC) + 1, STRIDE_SEC):
        end = start + WINDOW_SEC

        # A window is positive when any target event overlaps it
        overlap = (
            (events["start_sec"] < end) &
            (events["end_sec"] > start)
        ).any()

        label = int(overlap)

        rows.append({
            "subject_id": subject_id,
            "audio_path": str(wav.resolve()),
            "start_sec": start,
            "end_sec": end,
            "label": label
        })

        if label:
            positive += 1
        else:
            negative += 1

    print(
        f"{subject_id}: "
        f"{positive} positive, {negative} negative windows"
    )

if not rows:
    raise RuntimeError("No usable recordings found. Check DATASET path.")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)

print("\nTRAINING MANIFEST CREATED")
print("Saved to:", OUTPUT.resolve())
print("Total windows:", len(df))
print("\nLabel counts:")
print(df["label"].value_counts())
print("\nFirst five rows:")
print(df.head())
