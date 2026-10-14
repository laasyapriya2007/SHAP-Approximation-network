from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "shap"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("TRAINING SHAP APPROXIMATOR")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

X_train = pd.read_csv(
    SHAP_DIR / "X_train_shap.csv"
)

y_train = pd.read_csv(
    SHAP_DIR / "shap_train_values.csv"
)

print(f"\nX Shape : {X_train.shape}")
print(f"Y Shape : {y_train.shape}")

X_train = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
)

dataset = TensorDataset(
    X_train,
    y_train
)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

INPUT_DIM = X_train.shape[1]
OUTPUT_DIM = y_train.shape[1]

class SHAPApproximator(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                INPUT_DIM,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                32,
                OUTPUT_DIM
            )
        )

    def forward(
        self,
        x
    ):
        return self.net(x)

model = SHAPApproximator().to(
    DEVICE
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 100

print("\nTraining...\n")

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch "
        f"{epoch+1:3d}/{EPOCHS} "
        f"Loss: "
        f"{epoch_loss/len(loader):.6f}"
    )

torch.save(
    model.state_dict(),
    CHECKPOINT_DIR
    / "shap_tinymlp_weights.pth"
)

print("\nModel saved")

print(
    CHECKPOINT_DIR
    / "shap_tinymlp_weights.pth"
)
