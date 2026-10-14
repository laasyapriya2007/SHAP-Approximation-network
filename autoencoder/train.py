from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "shap"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("TRAINING SHAP AUTOENCODER")
print("=" * 70)

X_train = pd.read_csv(
    SHAP_DIR
    / "normal_shap_train_values.csv"
)

print(
    f"\nTraining Samples: "
    f"{len(X_train)}"
)

X_train = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

dataset = TensorDataset(
    X_train,
    X_train
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

INPUT_DIM = X_train.shape[1]

class Autoencoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8)
        )

        self.decoder = nn.Sequential(

            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, INPUT_DIM)
        )

    def forward(
        self,
        x
    ):

        z = self.encoder(x)

        out = self.decoder(z)

        return out

model = Autoencoder().to(
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

    for X_batch, _ in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        reconstructed = model(
            X_batch
        )

        loss = criterion(
            reconstructed,
            X_batch
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
    / "shap_autoencoder.pth"
)

print("\nModel Saved")

print(
    CHECKPOINT_DIR
    / "shap_autoencoder.pth"
)
