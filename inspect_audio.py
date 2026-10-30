
from pathlib import Path
import soundfile as sf
import pandas as pd

folder = Path.home() / "Downloads" / "APSAA" / "20211025C"

audio_path = folder / "20211025C.wav"
csv_path = folder / "20211025C_Annotations.csv"

info = sf.info(str(audio_path))
events = pd.read_csv(csv_path)

print("AUDIO INFORMATION")
print("Sample rate:", info.samplerate)
print("Channels:", info.channels)
print("Duration (seconds):", round(info.duration, 2))
print("Duration (hours):", round(info.duration / 3600, 2))

events["start_sec"] = pd.to_timedelta(
    events["Start_Time"]
).dt.total_seconds()

events["end_sec"] = (
    events["start_sec"] + events["Duration"]
)

print("\nEVENT TYPES:")
print(events["Event_Name"].value_counts())

print("\nFIRST APNEA EVENTS:")
print(
    events[
        events["Event_Name"].isin(
            ["Obstructive Apnea", "Hypopnea"]
        )
    ][["Event_Name", "start_sec", "end_sec"]].head()
)

print("\nLAST ANNOTATED EVENT ENDS AT:")
print(round(events["end_sec"].max(), 2), "seconds")
