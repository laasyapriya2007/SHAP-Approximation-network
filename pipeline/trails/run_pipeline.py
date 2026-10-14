from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

from sklearn.multioutput import MultiOutputRegressor

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
# IDS MODEL
# --------------------------------------------------
#

ids_model = joblib.load(
    CHECKPOINT_DIR
    / "lightgbm_weights.pkl"
)

#
# --------------------------------------------------
# SHAP APPROXIMATOR
# --------------------------------------------------
#

shap_model = joblib.load(
    CHECKPOINT_DIR
    / "shap_lightgbm_weights.pkl"
)

#
# --------------------------------------------------
# AUTOENCODER
# --------------------------------------------------
#

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

#
# --------------------------------------------------
# THRESHOLD
# --------------------------------------------------
#

normal_shap = pd.read_csv(
    SHAP_DIR
    / "normal_shap_test_values.csv"
)

normal_tensor = torch.tensor(
    normal_shap.values,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    reconstructed = autoencoder(
        normal_tensor
    )

normal_errors = torch.mean(

    (normal_tensor - reconstructed) ** 2,

    dim=1

).cpu().numpy()

threshold = (

    np.mean(normal_errors)

    +

    3 * np.std(normal_errors)

)

print(
    f"\nThreshold: "
    f"{threshold:.6f}"
)

#
# --------------------------------------------------
# PIPELINE
# --------------------------------------------------
#

results = []

print("\n")

for idx in range(
    len(X_pipeline)
):

    sample = X_pipeline.iloc[
        [idx]
    ]

    true_class = int(
        y_pipeline[idx]
    )

    start = time.perf_counter()

    #
    # IDS
    #

    pred_class = ids_model.predict(
        sample
    )[0]

    confidence = np.max(

        ids_model.predict_proba(
            sample
        )

    )

    #
    # SHAP APPROXIMATION
    #

    approx_shap = shap_model.predict(
        sample
    )

    #
    # AUTOENCODER
    #

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

    #
    # FUSION
    #

    fusion_score = (

        0.8
        *
        reconstruction_error

        +

        0.2
        *
        (1 - confidence)

    )

    decision = (

        "SUSPICIOUS"

        if reconstruction_error
        >
        threshold

        else

        "NORMAL"
    )

    end = time.perf_counter()

    inference_time = (
        end - start
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

    print("=" * 70)

    print(
        f"True Label          : {true_label}"
    )

    print(
        f"Predicted Label     : {pred_label}"
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
        f"Fusion Score        : "
        f"{fusion_score:.6f}"
    )

    print(
        f"Decision            : "
        f"{decision}"
    )

    print(
        f"Inference Time      : "
        f"{inference_time:.6f} sec"
    )

    results.append({

        "True_Label":
            true_label,

        "Predicted_Label":
            pred_label,

        "Confidence":
            confidence,

        "Reconstruction_Error":
            reconstruction_error,

        "Fusion_Score":
            fusion_score,

        "Decision":
            decision,

        "Inference_Time":
            inference_time
    })

results_df = pd.DataFrame(
    results
)

results_df.to_csv(

    SHAP_DIR
    / "pipeline_results.csv",

    index=False
)

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
