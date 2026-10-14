from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

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

X_test = pd.read_csv(
    SHAP_DIR
    / "shap_test_values.csv"
)

metadata = pd.read_csv(
    SHAP_DIR
    / "shap_test_metadata.csv"
)

INPUT_DIM = X_test.shape[1]

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

        return self.decoder(
            self.encoder(x)
        )

model = Autoencoder().to(
    DEVICE
)

model.load_state_dict(

    torch.load(

        CHECKPOINT_DIR
        / "shap_autoencoder.pth",

        map_location=DEVICE
    )
)

model.eval()

X_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    reconstructed = model(
        X_tensor
    )

errors = torch.mean(

    (X_tensor - reconstructed) ** 2,

    dim=1

).cpu().numpy()

results = metadata.copy()

results[
    "Reconstruction_Error"
] = errors

results.to_csv(
    SHAP_DIR
    / "reconstruction_errors.csv",
    index=False
)

print("=" * 70)
print("RECONSTRUCTION ERROR SUMMARY")
print("=" * 70)

print(
    results[
        "Reconstruction_Error"
    ].describe()
)

print("\nSaved:")

print(
    SHAP_DIR
    / "reconstruction_errors.csv"
)

print("\nTop 10 Highest Errors")

print(

    results.sort_values(

        by="Reconstruction_Error",

        ascending=False

    ).head(10)

)
