from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from model import SleepApneaCNN as OriginalCNN
from model_improved import SleepApneaCNN as ImprovedCNN


DATA_DIR = Path("data/processed")

ORIGINAL_MODEL = Path(
    "results/best_model.pt"
)

IMPROVED_MODEL = Path(
    "results_improved/best_model.pt"
)

VALIDATION_SUBJECT = "010"
TEST_SUBJECT = "014"


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def load_subject(subject):

    data = np.load(
        DATA_DIR / f"ucddb{subject}.npz"
    )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.int64
    )

    X = torch.from_numpy(
        X
    ).unsqueeze(1)

    return X, y


def get_probabilities(
    model,
    X,
    batch_size=64
):

    model.eval()

    probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size
        ):

            batch = X[
                start:start + batch_size
            ].to(DEVICE)

            logits = model(
                batch
            )

            probs = F.softmax(
                logits,
                dim=1
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy()
            )

    return np.array(
        probabilities
    )


print("=" * 60)
print("LOADING MODELS")
print("=" * 60)

original = OriginalCNN().to(
    DEVICE
)

original.load_state_dict(
    torch.load(
        ORIGINAL_MODEL,
        map_location=DEVICE
    )
)

improved = ImprovedCNN().to(
    DEVICE
)

improved.load_state_dict(
    torch.load(
        IMPROVED_MODEL,
        map_location=DEVICE
    )
)


print()
print("=" * 60)
print("LOADING VALIDATION")
print("=" * 60)

X_val, y_val = load_subject(
    VALIDATION_SUBJECT
)

original_val = get_probabilities(
    original,
    X_val
)

improved_val = get_probabilities(
    improved,
    X_val
)


# --------------------------------------------------
# SELECT ENSEMBLE WEIGHT USING VALIDATION AUC
# --------------------------------------------------

best_alpha = 0.0
best_auc = -1.0

print()
print("=" * 60)
print("ENSEMBLE WEIGHT SEARCH")
print("=" * 60)

for alpha in np.arange(
    0.0,
    1.01,
    0.05
):

    combined = (
        alpha * original_val
        +
        (1 - alpha) * improved_val
    )

    auc = roc_auc_score(
        y_val,
        combined
    )

    print(
        f"Original weight {alpha:.2f} | "
        f"Validation AUC {auc:.4f}"
    )

    if auc > best_auc:

        best_auc = auc
        best_alpha = alpha


print()
print(
    "Best original-model weight:",
    round(best_alpha, 2)
)

print(
    "Best validation AUC:",
    round(best_auc, 4)
)


combined_val = (
    best_alpha * original_val
    +
    (1 - best_alpha) * improved_val
)


# --------------------------------------------------
# SELECT THRESHOLD USING VALIDATION MCC
# --------------------------------------------------

best_threshold = 0.50
best_mcc = -1.0
best_accuracy = 0.0

print()
print("=" * 60)
print("THRESHOLD SEARCH")
print("=" * 60)

for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    predictions = (
        combined_val
        >= threshold
    ).astype(
        np.int64
    )

    mcc = matthews_corrcoef(
        y_val,
        predictions
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    if mcc > best_mcc:

        best_mcc = mcc
        best_threshold = threshold
        best_accuracy = accuracy


print(
    "Best threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "Validation MCC:",
    round(
        best_mcc,
        4
    )
)

print(
    "Validation accuracy:",
    round(
        best_accuracy,
        4
    )
)


# --------------------------------------------------
# TEST ON LOCKED SUBJECT 014
# --------------------------------------------------

print()
print("=" * 60)
print("LOCKED TEST SUBJECT 014")
print("=" * 60)

X_test, y_test = load_subject(
    TEST_SUBJECT
)

original_test = get_probabilities(
    original,
    X_test
)

improved_test = get_probabilities(
    improved,
    X_test
)

combined_test = (
    best_alpha * original_test
    +
    (1 - best_alpha) * improved_test
)

predictions = (
    combined_test
    >= best_threshold
).astype(
    np.int64
)


accuracy = accuracy_score(
    y_test,
    predictions
)

mcc = matthews_corrcoef(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    combined_test
)

matrix = confusion_matrix(
    y_test,
    predictions
)


print()
print(
    "Test Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Test MCC:",
    round(
        mcc,
        4
    )
)

print(
    "Test ROC AUC:",
    round(
        auc,
        4
    )
)

print()
print(
    "Confusion Matrix:"
)

print(
    matrix
)

print()
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Non-apnea",
            "Apnea"
        ],
        zero_division=0
    )
)