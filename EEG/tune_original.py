from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from model import SleepApneaCNN


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_DIR = Path("data/processed")

MODEL_PATH = Path(
    "results/best_model.pt"
)

VALIDATION_SUBJECT = "010"
TEST_SUBJECT = "014"

BATCH_SIZE = 64


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# --------------------------------------------------
# LOAD SUBJECT
# --------------------------------------------------

def load_subject(subject):

    path = (
        DATA_DIR
        / f"ucddb{subject}.npz"
    )

    data = np.load(
        path
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


# --------------------------------------------------
# MODEL PROBABILITIES
# --------------------------------------------------

def get_probabilities(
    model,
    X
):

    model.eval()

    probabilities = []

    softmax = nn.Softmax(
        dim=1
    )

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE
        ):

            batch = X[
                start:start + BATCH_SIZE
            ].to(
                DEVICE
            )

            logits = model(
                batch
            )

            probs = softmax(
                logits
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy()
            )

    return np.array(
        probabilities
    )


# --------------------------------------------------
# TEMPORAL SMOOTHING
# --------------------------------------------------

def smooth_probabilities(
    probabilities,
    window_size
):

    if window_size == 1:

        return probabilities.copy()

    padding = (
        window_size // 2
    )

    padded = np.pad(
        probabilities,
        (
            padding,
            padding
        ),
        mode="edge"
    )

    kernel = (
        np.ones(
            window_size
        )
        / window_size
    )

    smoothed = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

    return smoothed


# --------------------------------------------------
# FIND BEST SMOOTHING WINDOW
# --------------------------------------------------

def find_best_smoothing(
    labels,
    probabilities
):

    possible_windows = [
        1,
        3,
        5,
        7,
        9,
        11,
    ]

    best_window = 1
    best_auc = -1.0

    print()
    print("=" * 60)
    print("SMOOTHING SEARCH")
    print("=" * 60)

    for window in possible_windows:

        smoothed = smooth_probabilities(
            probabilities,
            window
        )

        auc = roc_auc_score(
            labels,
            smoothed
        )

        print(
            f"Window {window:2d} | "
            f"Validation AUC: {auc:.4f}"
        )

        if auc > best_auc:

            best_auc = auc
            best_window = window

    return (
        best_window,
        best_auc
    )


# --------------------------------------------------
# FIND BEST THRESHOLD
# --------------------------------------------------

def find_best_threshold(
    labels,
    probabilities
):

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
            probabilities
            >= threshold
        ).astype(
            np.int64
        )

        mcc = matthews_corrcoef(
            labels,
            predictions
        )

        accuracy = accuracy_score(
            labels,
            predictions
        )

        if mcc > best_mcc:

            best_mcc = mcc
            best_accuracy = accuracy
            best_threshold = threshold

    return (
        best_threshold,
        best_mcc,
        best_accuracy
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print()
    print("=" * 60)
    print("ORIGINAL MODEL POST-PROCESSING")
    print("=" * 60)

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = SleepApneaCNN().to(
        DEVICE
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    # --------------------------------------------------
    # VALIDATION SUBJECT
    # --------------------------------------------------

    print()
    print(
        "Loading validation subject:",
        VALIDATION_SUBJECT
    )

    X_validation, y_validation = (
        load_subject(
            VALIDATION_SUBJECT
        )
    )

    validation_probabilities = (
        get_probabilities(
            model,
            X_validation
        )
    )

    original_validation_auc = (
        roc_auc_score(
            y_validation,
            validation_probabilities
        )
    )

    print()
    print(
        "Original validation AUC:",
        round(
            original_validation_auc,
            4
        )
    )

    # --------------------------------------------------
    # FIND SMOOTHING
    # --------------------------------------------------

    (
        best_window,
        best_validation_auc
    ) = find_best_smoothing(
        y_validation,
        validation_probabilities
    )

    print()
    print(
        "Best smoothing window:",
        best_window
    )

    print(
        "Best validation AUC:",
        round(
            best_validation_auc,
            4
        )
    )

    validation_smoothed = (
        smooth_probabilities(
            validation_probabilities,
            best_window
        )
    )

    # --------------------------------------------------
    # FIND THRESHOLD
    # --------------------------------------------------

    (
        best_threshold,
        validation_mcc,
        validation_accuracy
    ) = find_best_threshold(
        y_validation,
        validation_smoothed
    )

    print()
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
            validation_mcc,
            4
        )
    )

    print(
        "Validation Accuracy:",
        round(
            validation_accuracy,
            4
        )
    )

    # --------------------------------------------------
    # TEST SUBJECT
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("LOCKED TEST SUBJECT 014")
    print("=" * 60)

    X_test, y_test = (
        load_subject(
            TEST_SUBJECT
        )
    )

    test_probabilities = (
        get_probabilities(
            model,
            X_test
        )
    )

    test_smoothed = (
        smooth_probabilities(
            test_probabilities,
            best_window
        )
    )

    predictions = (
        test_smoothed
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
        test_smoothed
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            0,
            1
        ]
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


if __name__ == "__main__":
    main()