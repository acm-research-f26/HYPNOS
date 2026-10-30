
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("training_windows.csv")

subjects = df["subject_id"].unique()

# 70% subjects for training, 30% temporarily held out
train_subjects, remaining_subjects = train_test_split(
    subjects,
    test_size=0.30,
    random_state=42
)

# Split remaining subjects equally into validation and test
val_subjects, test_subjects = train_test_split(
    remaining_subjects,
    test_size=0.50,
    random_state=42
)

train_df = df[df["subject_id"].isin(train_subjects)]
val_df = df[df["subject_id"].isin(val_subjects)]
test_df = df[df["subject_id"].isin(test_subjects)]

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("TRAINING:", len(train_df), "windows")
print("VALIDATION:", len(val_df), "windows")
print("TESTING:", len(test_df), "windows")

print("\nSUBJECT COUNTS:")
print("Train:", len(train_subjects))
print("Validation:", len(val_subjects))
print("Test:", len(test_subjects))

print("\nOVERLAP CHECK:")
print(
    "Train/Val:",
    len(set(train_subjects) & set(val_subjects))
)
print(
    "Train/Test:",
    len(set(train_subjects) & set(test_subjects))
)
print(
    "Val/Test:",
    len(set(val_subjects) & set(test_subjects))
)

print("\nLABEL DISTRIBUTION:")
for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(name, data["label"].value_counts().to_dict())
