from pathlib import Path
import json

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

from model import SleepApneaCNN


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_DIR = Path(
    "data/processed"
)

RESULTS_DIR = Path(
    "results_v3"
)

TRAIN_SUBJECTS = [
    "002",
    "003",
    "005",
    "006",
    "007",
    "012",
    "023",
    "028",
]

VALIDATION_SUBJECTS = [
    "010",
]

TEST_SUBJECTS = [
    "014",
]

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.0005

RANDOM_SEED = 42


# --------------------------------------------------
# RANDOM SEEDS
# --------------------------------------------------

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


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
# SUBJECT BALANCING
# --------------------------------------------------

def balance_subject(X, y, random_generator):
    """
    Randomly undersamples the majority class.

    Used for training and validation.
    """

    apnea_indices = np.where(
        y == 1
    )[0]

    non_apnea_indices = np.where(
        y == 0
    )[0]

    minority_count = min(
        len(apnea_indices),
        len(non_apnea_indices)
    )

    if minority_count == 0:

        print(
            "WARNING: subject has only one class."
        )

        return X, y

    apnea_sample = (
        random_generator.choice(
            apnea_indices,
            size=minority_count,
            replace=False
        )
    )

    non_apnea_sample = (
        random_generator.choice(
            non_apnea_indices,
            size=minority_count,
            replace=False
        )
    )

    selected_indices = np.concatenate(
        [
            apnea_sample,
            non_apnea_sample
        ]
    )

    random_generator.shuffle(
        selected_indices
    )

    return (
        X[selected_indices],
        y[selected_indices]
    )


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_subjects(
    subjects,
    balance=False
):

    all_X = []
    all_y = []

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    for subject in subjects:

        path = (
            DATA_DIR
            / f"ucddb{subject}.npz"
        )

        print(
            f"Loading {path}"
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

        apnea_count = int(
            np.sum(y == 1)
        )

        non_apnea_count = int(
            np.sum(y == 0)
        )

        print(
            f"  Segments: {len(y)}"
        )

        print(
            f"  Apnea: {apnea_count}"
        )

        print(
            f"  Non-apnea: {non_apnea_count}"
        )

        if balance:

            X, y = balance_subject(
                X,
                y,
                rng
            )

            print(
                f"  Balanced: {len(y)} segments"
            )

        all_X.append(
            X
        )

        all_y.append(
            y
        )

    X = np.concatenate(
        all_X,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    return X, y


# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------

def make_loader(
    X,
    y,
    shuffle
):

    X_tensor = torch.from_numpy(
        X
    ).unsqueeze(
        1
    )

    y_tensor = torch.from_numpy(
        y
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

    return loader


# --------------------------------------------------
# TRAIN ONE EPOCH
# --------------------------------------------------

def train_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in loader:

        X = X.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            X
        )

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * X.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_correct += (
            predictions == y
        ).sum().item()

        total_samples += (
            X.size(0)
        )

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy
    )


# --------------------------------------------------
# EVALUATE LOSS / ACCURACY
# --------------------------------------------------

def validation_epoch(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )

            logits = model(
                X
            )

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item()
                * X.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_correct += (
                predictions == y
            ).sum().item()

            total_samples += (
                X.size(0)
            )

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy
    )


# --------------------------------------------------
# GET MODEL PREDICTIONS
# --------------------------------------------------

def test_model(
    model,
    loader
):

    model.eval()

    true_labels = []
    probabilities = []
    predictions = []

    softmax = nn.Softmax(
        dim=1
    )

    with torch.no_grad():

        for X, y in loader:

            X = X.to(
                DEVICE
            )

            logits = model(
                X
            )

            probability = softmax(
                logits
            )

            predicted = torch.argmax(
                probability,
                dim=1
            )

            true_labels.extend(
                y.numpy()
            )

            probabilities.extend(
                probability[
                    :, 1
                ]
                .cpu()
                .numpy()
            )

            predictions.extend(
                predicted
                .cpu()
                .numpy()
            )

    true_labels = np.array(
        true_labels
    )

    probabilities = np.array(
        probabilities
    )

    predictions = np.array(
        predictions
    )

    return (
        true_labels,
        probabilities,
        predictions
    )


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

def plot_training_history(
    train_losses,
    validation_losses,
    train_accuracies,
    validation_accuracies,
    validation_aucs
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # LOSS

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        train_losses,
        label="Training"
    )

    plt.plot(
        validation_losses,
        label="Validation"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "loss.png",
        dpi=200
    )

    plt.close()

    # ACCURACY

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        train_accuracies,
        label="Training"
    )

    plt.plot(
        validation_accuracies,
        label="Validation"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "accuracy.png",
        dpi=200
    )

    plt.close()

    # VALIDATION AUC

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        validation_aucs
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "ROC AUC"
    )

    plt.title(
        "Validation ROC AUC"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "validation_auc.png",
        dpi=200
    )

    plt.close()


def plot_confusion_matrix(
    matrix
):

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
        cmap="Blues"
    )

    plt.title(
        "Sleep Apnea Confusion Matrix"
    )

    plt.colorbar()

    classes = [
        "Non-apnea",
        "Apnea"
    ]

    plt.xticks(
        [0, 1],
        classes
    )

    plt.yticks(
        [0, 1],
        classes
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    for row in range(2):

        for column in range(2):

            plt.text(
                column,
                row,
                matrix[
                    row,
                    column
                ],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "confusion_matrix.png",
        dpi=200
    )

    plt.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("HYPNOS EEG SLEEP APNEA MODEL V3")
    print("=" * 60)

    print()
    print(
        "Device:",
        DEVICE
    )

    print()
    print(
        "Training subjects:",
        TRAIN_SUBJECTS
    )

    print(
        "Validation subjects:",
        VALIDATION_SUBJECTS
    )

    print(
        "Test subjects:",
        TEST_SUBJECTS
    )

    # --------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    X_train, y_train = load_subjects(
        TRAIN_SUBJECTS,
        balance=True
    )

    # --------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING VALIDATION DATA")
    print("=" * 60)

    X_validation, y_validation = load_subjects(
        VALIDATION_SUBJECTS,
        balance=True
    )

    # --------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)

    X_test, y_test = load_subjects(
        TEST_SUBJECTS,
        balance=False
    )

    print()
    print(
        "Final shapes:"
    )

    print(
        "Train:",
        X_train.shape,
        y_train.shape
    )

    print(
        "Validation:",
        X_validation.shape,
        y_validation.shape
    )

    print(
        "Test:",
        X_test.shape,
        y_test.shape
    )

    # --------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------

    train_loader = make_loader(
        X_train,
        y_train,
        shuffle=True
    )

    validation_loader = make_loader(
        X_validation,
        y_validation,
        shuffle=False
    )

    test_loader = make_loader(
        X_test,
        y_test,
        shuffle=False
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = SleepApneaCNN().to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(
            0.9,
            0.999
        )
    )

    print()
    print(model)

    # --------------------------------------------------
    # TRAINING STORAGE
    # --------------------------------------------------

    train_losses = []
    validation_losses = []

    train_accuracies = []
    validation_accuracies = []

    validation_aucs = []

    best_validation_auc = -1.0

    best_model_path = (
        RESULTS_DIR
        / "best_model.pt"
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # TRAIN

        train_loss, train_accuracy = (
            train_epoch(
                model,
                train_loader,
                optimizer,
                criterion
            )
        )

        # VALIDATION LOSS + ACCURACY

        (
            validation_loss,
            validation_accuracy
        ) = validation_epoch(
            model,
            validation_loader,
            criterion
        )

        # VALIDATION AUC

        (
            validation_true,
            validation_probabilities,
            validation_predictions
        ) = test_model(
            model,
            validation_loader
        )

        if len(
            np.unique(
                validation_true
            )
        ) == 2:

            validation_auc = roc_auc_score(
                validation_true,
                validation_probabilities
            )

        else:

            validation_auc = 0.0

        # STORE HISTORY

        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        validation_accuracies.append(
            validation_accuracy
        )

        validation_aucs.append(
            validation_auc
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {validation_loss:.4f} | "
            f"Val acc: {validation_accuracy:.4f} | "
            f"Val AUC: {validation_auc:.4f}"
        )

        # SAVE MODEL BASED ON VALIDATION AUC

        if (
            validation_auc
            > best_validation_auc
        ):

            best_validation_auc = (
                validation_auc
            )

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                "  Saved new best model."
            )

    # --------------------------------------------------
    # SAVE TRAINING PLOTS
    # --------------------------------------------------

    plot_training_history(
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies,
        validation_aucs
    )

    # --------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------

    print()
    print(
        "Best validation AUC:",
        round(
            best_validation_auc,
            4
        )
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE
        )
    )

    # --------------------------------------------------
    # THRESHOLD SEARCH ON VALIDATION SET
    # --------------------------------------------------
    # The default 0.5 cutoff isn't guaranteed to be optimal,
    # especially once the model is tested against the
    # natural (unbalanced) class distribution. We sweep
    # thresholds here and pick the one that maximizes MCC
    # on validation data.

    print()
    print("=" * 60)
    print("THRESHOLD SEARCH")
    print("=" * 60)

    (
        validation_true,
        validation_probabilities,
        _
    ) = test_model(
        model,
        validation_loader
    )

    best_threshold = 0.50
    best_threshold_mcc = -1.0

    for threshold in np.arange(
        0.10,
        0.91,
        0.01
    ):

        thresholded = (
            validation_probabilities
            >= threshold
        ).astype(
            np.int64
        )

        threshold_mcc = matthews_corrcoef(
            validation_true,
            thresholded
        )

        if threshold_mcc > best_threshold_mcc:

            best_threshold_mcc = threshold_mcc
            best_threshold = threshold

    print(
        "Best threshold:",
        round(
            best_threshold,
            2
        )
    )

    print(
        "Validation MCC at best threshold:",
        round(
            best_threshold_mcc,
            4
        )
    )

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TESTING LOCKED SUBJECT 014")
    print("=" * 60)

    (
        true_labels,
        probabilities,
        _
    ) = test_model(
        model,
        test_loader
    )

    predictions = (
        probabilities
        >= best_threshold
    ).astype(
        np.int64
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    mcc = matthews_corrcoef(
        true_labels,
        predictions
    )

    if len(
        np.unique(
            true_labels
        )
    ) == 2:

        auc = roc_auc_score(
            true_labels,
            probabilities
        )

    else:

        auc = float(
            "nan"
        )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[
            0,
            1
        ]
    )

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

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
        "Classification Report:"
    )

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=[
                "Non-apnea",
                "Apnea"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    metrics = {

        "training_subjects":
            TRAIN_SUBJECTS,

        "validation_subjects":
            VALIDATION_SUBJECTS,

        "test_subjects":
            TEST_SUBJECTS,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "best_validation_auc":
            float(
                best_validation_auc
            ),

        "best_threshold":
            float(
                best_threshold
            ),

        "accuracy":
            float(
                accuracy
            ),

        "mcc":
            float(
                mcc
            ),

        "roc_auc":
            float(
                auc
            ),

        "confusion_matrix":
            matrix.tolist()
    }

    with open(
        RESULTS_DIR
        / "metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    plot_confusion_matrix(
        matrix
    )

    # --------------------------------------------------
    # FINISHED
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    print()
    print(
        "Model saved to:",
        best_model_path
    )

    print(
        "Metrics saved to:",
        RESULTS_DIR
        / "metrics.json"
    )

    print(
        "Plots saved to:",
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()