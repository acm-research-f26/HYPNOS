
import pandas as pd
from pathlib import Path

csv_path = Path.home() / "Downloads" / "APSAA" / "20211025C" / "20211025C_Annotations.csv"

if not csv_path.exists():
    print("CSV not found at:", csv_path)
    print("We need to locate the extracted APSAA folder.")
else:
    df = pd.read_csv(csv_path)

    print("COLUMN NAMES:")
    print(df.columns.tolist())

    print("\nFIRST 10 ROWS:")
    print(df.head(10))

    print("\nTOTAL ANNOTATIONS:", len(df))
