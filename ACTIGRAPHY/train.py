
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from ACTIGRAPHY.model import ActigraphyCNN


def train_model(X_train, y_train, X_val, y_val):
    # Use GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Convert datasets into PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Prepare batches of training examples
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64
    )

    # Initialize our 1D CNN
    model = ActigraphyCNN().to(device)

    # Loss function measures prediction errors
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer updates weights so the model learns
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    best_val_loss = float("inf")

    for epoch in range(10):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        # Check how well the model works on validation data
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * len(X_batch)

        train_loss /= len(X_train)
        val_loss /= len(X_val)

        print(
            f"Epoch {epoch + 1}/10 | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        # Save the best-performing model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            output_path = Path(__file__).parent / "models"
            output_path.mkdir(exist_ok=True)

            torch.save(
                model.state_dict(),
                output_path / "actigraphy_cnn.pth"
            )

    print("Training complete!")
    print("Best model saved to ACTIGRAPHY/models/")
