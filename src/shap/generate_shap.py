from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

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

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

print("=" * 70)
print("GENERATING SHAP FINGERPRINTS")
print("=" * 70)

print("\nLoading dataset...")

X = pd.read_csv(
    SHAP_DIR / "X_shap.csv"
)

y = pd.read_csv(
    SHAP_DIR / "y_shap.csv"
)

print(f"X Shape : {X.shape}")
print(f"y Shape : {y.shape}")

print("\nLoading LightGBM model...")

model = joblib.load(
    CHECKPOINT_DIR
    / "lightgbm_weights.pkl"
)

target_encoder = joblib.load(
    PROCESSED_DIR
    / "target_encoder.pkl"
)

print("Model loaded")

print("\nGenerating predictions...")

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence = np.max(
    probabilities,
    axis=1
)

print("\nCreating TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("Generating exact SHAP values...")
print("This may take several minutes...\n")

shap_values = explainer.shap_values(X)

print("SHAP generation completed")

print("\nProcessing multiclass SHAP output...")

if isinstance(shap_values, list):

    print("Detected OLD SHAP format")

    predicted_shap = np.zeros(
        (
            len(X),
            X.shape[1]
        )
    )

    for i in range(len(X)):

        predicted_class = predictions[i]

        predicted_shap[i] = (
            shap_values[
                predicted_class
            ][i]
        )

else:

    print("Detected NEW SHAP format")

    predicted_shap = np.zeros(
        (
            len(X),
            X.shape[1]
        )
    )

    for i in range(len(X)):

        predicted_class = predictions[i]

        predicted_shap[i] = (
            shap_values[
                i,
                :,
                predicted_class
            ]
        )

print(
    f"\nFinal SHAP Shape: "
    f"{predicted_shap.shape}"
)

print("\nCreating SHAP dataframe...")

shap_df = pd.DataFrame(
    predicted_shap,
    columns=X.columns
)

print("Saving SHAP fingerprints...")

shap_df.to_csv(
    SHAP_DIR / "shap_values.csv",
    index=False
)

print("Creating metadata...")

true_labels = target_encoder.inverse_transform(
    y["Attack_type"]
)

pred_labels = target_encoder.inverse_transform(
    predictions
)

metadata_df = pd.DataFrame(
    {
        "True_Label": true_labels,
        "Predicted_Label": pred_labels,
        "Confidence": confidence
    }
)

metadata_df.to_csv(
    SHAP_DIR / "shap_metadata.csv",
    index=False
)

print("\nSaved:")

print(
    SHAP_DIR / "shap_values.csv"
)

print(
    SHAP_DIR / "shap_metadata.csv"
)

print("\nSummary")

print(
    f"Samples  : {len(X):,}"
)

print(
    f"Features : {X.shape[1]}"
)

print(
    f"Classes  : {len(target_encoder.classes_)}"
)

print("\n" + "=" * 70)
print("SHAP GENERATION COMPLETED")
print("=" * 70)
