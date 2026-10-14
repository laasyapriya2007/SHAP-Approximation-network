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
# BUILD 5 SAMPLES PER CLASS
# --------------------------------------------------
#

df = pd.concat(
    [
        X_test,
        y_test
    ],
    axis=1
)

df = df.reset_index()

sampled_dfs = []

for cls in sorted(
    df["Attack_type"].unique()
):

    class_df = df[
        df["Attack_type"] == cls
    ]

    sampled = class_df.sample(
        n=min(5, len(class_df)),
        random_state=42
    )

    sampled_dfs.append(
        sampled
    )

pipeline_df = pd.concat(
    sampled_dfs,
    ignore_index=True
)

pipeline_df = pipeline_df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)

print(
    f"\nSelected Samples: "
    f"{len(pipeline_df)}"
)

#
# --------------------------------------------------
# EXTRACT
# --------------------------------------------------
#

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

exact_pipeline_shap = exact_shap.iloc[
    original_indices
]

#
# --------------------------------------------------
# LOAD IDS
# --------------------------------------------------
#

print("\nLoading IDS...")

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
# PIPELINE
# --------------------------------------------------
#

results = []

approx_shap_list = []

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

    ids_time = (
        time.perf_counter()
        -
        ids_start
    )

    #
    # SHAP APPROXIMATION
    #

    shap_start = time.perf_counter()

    approx_shap = shap_model.predict(
        sample
    )

    shap_time = (
        time.perf_counter()
        -
        shap_start
    )

    approx_shap_list.append(
        approx_shap.flatten()
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

    ae_time = (
        time.perf_counter()
        -
        ae_start
    )

    total_time = (
        ids_time
        +
        shap_time
        +
        ae_time
    )

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

    results.append({

        "Original_Index":
            original_indices[idx],

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
# SAVE EXACT SHAP
#

exact_pipeline_shap.to_csv(

    SHAP_DIR
    / "pipeline_exact_shap.csv",

    index=False
)

#
# SAVE APPROX SHAP
#

pd.DataFrame(
    approx_shap_list,
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
    f"\nAverage IDS Time         : "
    f"{results_df['IDS_Time'].mean():.6f}"
)

print(
    f"Average SHAP Time        : "
    f"{results_df['SHAP_Time'].mean():.6f}"
)

print(
    f"Average Autoencoder Time : "
    f"{results_df['Autoencoder_Time'].mean():.6f}"
)

print(
    f"Average Pipeline Time    : "
    f"{results_df['Total_Pipeline_Time'].mean():.6f}"
)
