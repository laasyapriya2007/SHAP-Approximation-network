from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "shap"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
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
print("RUNNING IDS + SHAP + AUTOENCODER PIPELINE")
print("=" * 70)

#
# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
#

X_test = pd.read_csv(
    SHAP_DIR / "X_test_shap.csv"
)

y_test = pd.read_csv(
    SHAP_DIR / "y_test_shap.csv"
)

exact_shap = pd.read_csv(
    SHAP_DIR / "shap_test_values.csv"
)

target_encoder = joblib.load(
    PROCESSED_DIR
    / "target_encoder.pkl"
)

print(f"\nX Shape      : {X_test.shape}")
print(f"y Shape      : {y_test.shape}")
print(f"SHAP Shape   : {exact_shap.shape}")

#
# --------------------------------------------------
# USE ENTIRE TEST SHAP DATASET
# --------------------------------------------------
#

pipeline_df = pd.concat(
    [
        X_test,
        y_test
    ],
    axis=1
).reset_index()

print(
    f"\nUsing Entire Dataset: "
    f"{len(pipeline_df)} samples"
)

original_indices = pipeline_df[
    "index"
].values

X_pipeline = pipeline_df.drop(
    columns=[
        "index",
        "Attack_type"
    ]
)

y_pipeline = pipeline_df[
    "Attack_type"
].values

exact_pipeline_shap = exact_shap.copy()

#
# --------------------------------------------------
# LOAD IDS
# --------------------------------------------------
#

print("\nLoading LightGBM Model...")

ids_model = joblib.load(
    CHECKPOINT_DIR
    / "lightgbm_weights.pkl"
)

#
# --------------------------------------------------
# LOAD SHAP APPROXIMATOR
# --------------------------------------------------
#

print("Loading SHAP Approximator... (LightGBM Regressor)")

shap_model = joblib.load(
    CHECKPOINT_DIR
    / "shap_lightgbm_weights.pkl"
)

#
# --------------------------------------------------
# LOAD AUTOENCODER
# --------------------------------------------------
#

print("Loading Autoencoder...")

INPUT_DIM = exact_pipeline_shap.shape[1]

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

autoencoder = Autoencoder().to(
    DEVICE
)

autoencoder.load_state_dict(

    torch.load(
        CHECKPOINT_DIR
        / "shap_autoencoder.pth",
        map_location=DEVICE
    )
)

autoencoder.eval()

print("Models Loaded")

#
# --------------------------------------------------
# IDS INFERENCE (ALL SAMPLES)
# --------------------------------------------------
#

print("\nRunning IDS...")

ids_start = time.perf_counter()

pred_classes = ids_model.predict(
    X_pipeline
)

probabilities = ids_model.predict_proba(
    X_pipeline
)

confidences = np.max(
    probabilities,
    axis=1
)

ids_end = time.perf_counter()

total_ids_time = (
    ids_end - ids_start
)

avg_ids_time = (
    total_ids_time
    / len(X_pipeline)
)

print(
    f"IDS Completed "
    f"({total_ids_time:.4f} sec)"
)

#
# --------------------------------------------------
# SHAP APPROXIMATION (ALL SAMPLES)
# --------------------------------------------------
#

print("\nRunning SHAP Approximator...")

shap_start = time.perf_counter()

approx_shap_all = shap_model.predict(
    X_pipeline
)

shap_end = time.perf_counter()

total_shap_time = (
    shap_end - shap_start
)

avg_shap_time = (
    total_shap_time
    / len(X_pipeline)
)

print(
    f"SHAP Completed "
    f"({total_shap_time:.4f} sec)"
)

#
# --------------------------------------------------
# AUTOENCODER (ALL SAMPLES)
# --------------------------------------------------
#

print("\nRunning Autoencoder...")

ae_start = time.perf_counter()

shap_tensor = torch.tensor(
    approx_shap_all,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    reconstructed = autoencoder(
        shap_tensor
    )

reconstruction_errors = torch.mean(

    (shap_tensor - reconstructed) ** 2,

    dim=1

).cpu().numpy()

ae_end = time.perf_counter()

total_ae_time = (
    ae_end - ae_start
)

avg_ae_time = (
    total_ae_time
    / len(X_pipeline)
)

print(
    f"Autoencoder Completed "
    f"({total_ae_time:.4f} sec)"
)

#
# --------------------------------------------------
# BUILD RESULTS
# --------------------------------------------------
#

print("\nBuilding Results...")

results = []

for idx in range(
    len(X_pipeline)
):

    true_label = (
        target_encoder
        .inverse_transform(
            [int(y_pipeline[idx])]
        )[0]
    )

    pred_label = (
        target_encoder
        .inverse_transform(
            [int(pred_classes[idx])]
        )[0]
    )

    results.append({

        "Original_Index":
            original_indices[idx],

        "True_Label":
            true_label,

        "Predicted_Label":
            pred_label,

        "Confidence":
            confidences[idx],

        "Reconstruction_Error":
            reconstruction_errors[idx],

        "IDS_Time":
            avg_ids_time,

        "SHAP_Time":
            avg_shap_time,

        "Autoencoder_Time":
            avg_ae_time,

        "Total_Pipeline_Time":
            (
                avg_ids_time
                +
                avg_shap_time
                +
                avg_ae_time
            )
    })

results_df = pd.DataFrame(
    results
)

#
# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
#

results_df.to_csv(

    SHAP_DIR
    / "pipeline_results.csv",

    index=False
)

exact_pipeline_shap.to_csv(

    SHAP_DIR
    / "pipeline_exact_shap.csv",

    index=False
)

pd.DataFrame(

    approx_shap_all,

    columns=exact_pipeline_shap.columns

).to_csv(

    SHAP_DIR
    / "pipeline_approx_shap.csv",

    index=False
)

print("\n" + "=" * 70)
print("PIPELINE COMPLETED")
print("=" * 70)

print("\nSaved:")

print(
    SHAP_DIR
    / "pipeline_results.csv"
)

print(
    SHAP_DIR
    / "pipeline_exact_shap.csv"
)

print(
    SHAP_DIR
    / "pipeline_approx_shap.csv"
)

print("\nSummary")

print(
    f"\nTotal IDS Time         : "
    f"{total_ids_time:.6f} sec"
)

print(
    f"Total SHAP Time        : "
    f"{total_shap_time:.6f} sec"
)

print(
    f"Total Autoencoder Time : "
    f"{total_ae_time:.6f} sec"
)

print(
    f"\nAverage IDS Time       : "
    f"{avg_ids_time:.8f} sec/sample"
)

print(
    f"Average SHAP Time      : "
    f"{avg_shap_time:.8f} sec/sample"
)

print(
    f"Average AE Time        : "
    f"{avg_ae_time:.8f} sec/sample"
)
