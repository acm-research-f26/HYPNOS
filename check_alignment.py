
from pathlib import Path
import pandas as pd
import soundfile as sf

folder = Path.home() / "Downloads" / "APSAA" / "20211025C"

audio_path = folder / "20211025C.wav"
annotations_path = folder / "20211025C_Annotations.csv"

audio = sf.info(str(audio_path))
annotations = pd.read_csv(annotations_path)

annotations["start_sec"] = pd.to_timedelta(
    annotations["Start_Time"]
).dt.total_seconds()

annotations["end_sec"] = (
    annotations["start_sec"] + annotations["Duration"]
)

print("Audio duration:", round(audio.duration, 2), "seconds")
print("First annotation:", annotations["start_sec"].min())
print("Last annotation:", annotations["end_sec"].max())

out_of_range = annotations[
    (annotations["start_sec"] < 0) |
    (annotations["end_sec"] > audio.duration)
]

print("Annotations outside recording:", len(out_of_range))

if len(out_of_range) == 0:
    print("PASS: All annotated events fit within the WAV duration.")
else:
    print("WARNING: Some annotations fall outside the WAV duration.")
