from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

SHAP_DIR = (
    DATA_DIR
    / "shap"
)

PROCESSED_DIR = (
    DATA_DIR
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
print("RUNNING FULL IDS PIPELINE")
print("=" * 70)

#
# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
#

X_test = pd.read_csv(
    PROCESSED_DIR / "X_test.csv"
)

y_test = pd.read_csv(
    PROCESSED_DIR / "y_test.csv"
)

target_encoder = joblib.load(
    PROCESSED_DIR
    / "target_encoder.pkl"
)

#
# --------------------------------------------------
# SAMPLE 5 PER CLASS
# --------------------------------------------------
#

df = pd.concat(
    [X_test, y_test],
    axis=1
)

sampled = []

for cls in sorted(
    df["Attack_type"].unique()
):

    cls_df = df[
        df["Attack_type"] == cls
    ]

    sampled.append(

        cls_df.sample(
            n=min(5, len(cls_df)),
            random_state=42
        )
    )

pipeline_df = pd.concat(
    sampled,
    ignore_index=True
)

pipeline_df = pipeline_df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)

X_pipeline = pipeline_df.drop(
    columns=["Attack_type"]
)

y_pipeline = pipeline_df[
    "Attack_type"
].values

print(
    f"\nSamples Selected: "
    f"{len(X_pipeline)}"
)

#
# --------------------------------------------------
# LOAD IDS MODEL
# --------------------------------------------------
#

print("\nLoading IDS Model...")

ids_model = joblib.load(
    CHECKPOINT_DIR
    / "lightgbm_weights.pkl"
)

#
# --------------------------------------------------
# LOAD SHAP APPROXIMATOR
# --------------------------------------------------
#

print("Loading SHAP Approximator...")

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

INPUT_DIM = 47

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

print("\nModels Loaded Successfully")

#
# --------------------------------------------------
# PIPELINE
# --------------------------------------------------
#

results = []

print("\nRunning Pipeline...\n")

for idx in range(
    len(X_pipeline)
):

    sample = X_pipeline.iloc[
        [idx]
    ]

    true_class = int(
        y_pipeline[idx]
    )

    #
    # IDS
    #

    ids_start = time.perf_counter()

    pred_class = ids_model.predict(
        sample
    )[0]

    confidence = np.max(

        ids_model.predict_proba(
            sample
        )

    )

    ids_end = time.perf_counter()

    ids_time = (
        ids_end - ids_start
    )

    #
    # SHAP APPROXIMATION
    #

    shap_start = time.perf_counter()

    approx_shap = shap_model.predict(
        sample
    )

    shap_end = time.perf_counter()

    shap_time = (
        shap_end - shap_start
    )

    #
    # AUTOENCODER
    #

    ae_start = time.perf_counter()

    shap_tensor = torch.tensor(
        approx_shap,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        reconstructed = autoencoder(
            shap_tensor
        )

    reconstruction_error = torch.mean(

        (shap_tensor - reconstructed) ** 2

    ).item()

    ae_end = time.perf_counter()

    ae_time = (
        ae_end - ae_start
    )

    #
    # TOTAL
    #

    total_time = (

        ids_time
        +
        shap_time
        +
        ae_time
    )

    #
    # LABEL DECODING
    #

    true_label = (
        target_encoder
        .inverse_transform(
            [true_class]
        )[0]
    )

    pred_label = (
        target_encoder
        .inverse_transform(
            [pred_class]
        )[0]
    )

    #
    # PRINT
    #

    print("=" * 70)

    print(
        f"True Label          : "
        f"{true_label}"
    )

    print(
        f"Predicted Label     : "
        f"{pred_label}"
    )

    print(
        f"Confidence          : "
        f"{confidence:.6f}"
    )

    print(
        f"Reconstruction Error: "
        f"{reconstruction_error:.6f}"
    )

    print(
        f"IDS Time            : "
        f"{ids_time:.6f} sec"
    )

    print(
        f"SHAP Time           : "
        f"{shap_time:.6f} sec"
    )

    print(
        f"Autoencoder Time    : "
        f"{ae_time:.6f} sec"
    )

    print(
        f"Total Pipeline Time : "
        f"{total_time:.6f} sec"
    )

    #
    # SAVE
    #

    results.append({

        "True_Label":
            true_label,

        "Predicted_Label":
            pred_label,

        "Confidence":
            confidence,

        "Reconstruction_Error":
            reconstruction_error,

        "IDS_Time":
            ids_time,

        "SHAP_Time":
            shap_time,

        "Autoencoder_Time":
            ae_time,

        "Total_Pipeline_Time":
            total_time
    })

#
# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
#

results_df = pd.DataFrame(
    results
)

results_df.to_csv(

    SHAP_DIR
    / "pipeline_results.csv",

    index=False
)

#
# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
#

print("\n" + "=" * 70)
print("PIPELINE COMPLETED")
print("=" * 70)

print(
    "\nSaved:"
)

print(
    SHAP_DIR
    / "pipeline_results.csv"
)

print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)

print(
    f"\nAverage IDS Time            : "
    f"{results_df['IDS_Time'].mean():.6f} sec"
)

print(
    f"Average SHAP Time           : "
    f"{results_df['SHAP_Time'].mean():.6f} sec"
)

print(
    f"Average Autoencoder Time    : "
    f"{results_df['Autoencoder_Time'].mean():.6f} sec"
)

print(
    f"Average Pipeline Time       : "
    f"{results_df['Total_Pipeline_Time'].mean():.6f} sec"
)

print(
    f"\nMedian IDS Time             : "
    f"{results_df['IDS_Time'].median():.6f} sec"
)

print(
    f"Median SHAP Time            : "
    f"{results_df['SHAP_Time'].median():.6f} sec"
)

print(
    f"Median Autoencoder Time     : "
    f"{results_df['Autoencoder_Time'].median():.6f} sec"
)

print(
    f"Median Pipeline Time        : "
    f"{results_df['Total_Pipeline_Time'].median():.6f} sec"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
