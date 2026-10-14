from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import time

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

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
print("EVALUATING TINYMLP SHAP APPROXIMATOR")
print("=" * 70)

X_test = pd.read_csv(
    SHAP_DIR
    / "X_test_shap.csv"
)

y_test = pd.read_csv(
    SHAP_DIR
    / "shap_test_values.csv"
)

INPUT_DIM = X_test.shape[1]
OUTPUT_DIM = y_test.shape[1]

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

model_path = (
    CHECKPOINT_DIR
    / "shap_tinymlp_weights.pth"
)

model.load_state_dict(

    torch.load(
        model_path,
        map_location=DEVICE
    )
)

model.eval()

X_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
).to(DEVICE)

start = time.perf_counter()

with torch.no_grad():

    y_pred = model(
        X_tensor
    ).cpu().numpy()

end = time.perf_counter()

inference_time = (
    end - start
)

size_mb = (
    os.path.getsize(model_path)
    / (1024 * 1024)
)

y_true = y_test.values

mse = mean_squared_error(
    y_true,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_true,
    y_pred
)

r2 = r2_score(
    y_true,
    y_pred
)

dot = np.sum(
    y_true * y_pred,
    axis=1
)

norm_true = np.linalg.norm(
    y_true,
    axis=1
)

norm_pred = np.linalg.norm(
    y_pred,
    axis=1
)

cos_scores = (
    dot
    /
    (
        norm_true *
        norm_pred +
        1e-10
    )
)

avg_cos = np.mean(
    cos_scores
)

print("\nMetrics")
print("=" * 70)

print(
    f"MSE               : {mse:.8f}"
)

print(
    f"RMSE              : {rmse:.8f}"
)

print(
    f"MAE               : {mae:.8f}"
)

print(
    f"R2 Score          : {r2:.8f}"
)

print(
    f"Cosine Similarity : {avg_cos:.8f}"
)

print(
    f"Inference Time    : {inference_time:.6f} sec"
)

print(
    f"Model Size        : {size_mb:.4f} MB"
)

print("\n" + "=" * 70)
print("SINGLE SAMPLE")
print("=" * 70)

sample_idx = 0

actual_shap = y_true[
    sample_idx
]

predicted_shap = y_pred[
    sample_idx
]

sample_cos = (
    np.dot(
        actual_shap,
        predicted_shap
    )
    /
    (
        np.linalg.norm(
            actual_shap
        )
        *
        np.linalg.norm(
            predicted_shap
        )
        +
        1e-10
    )
)

print(
    f"Sample Index      : {sample_idx}"
)

print(
    f"Cosine Similarity : {sample_cos:.6f}"
)

print("\nActual SHAP")

print(
    np.round(
        actual_shap,
        6
    )
)

print("\nPredicted SHAP")

print(
    np.round(
        predicted_shap,
        6
    )
)

print("\nAbsolute Error")

print(
    np.round(
        np.abs(
            actual_shap
            -
            predicted_shap
        ),
        6
    )
)
