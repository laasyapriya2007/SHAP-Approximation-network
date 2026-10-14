from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_DIR = PROJECT_ROOT / "data" / "shap"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

print("=" * 70)
print("GENERATING TRAIN SHAP FINGERPRINTS")
print("=" * 70)

X = pd.read_csv(
    SHAP_DIR / "X_train_shap.csv"
)

y = pd.read_csv(
    SHAP_DIR / "y_train_shap.csv"
)

print(f"X Shape : {X.shape}")
print(f"y Shape : {y.shape}")

model = joblib.load(
    CHECKPOINT_DIR / "lightgbm_weights.pkl"
)

target_encoder = joblib.load(
    PROCESSED_DIR / "target_encoder.pkl"
)

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence = np.max(
    probabilities,
    axis=1
)

explainer = shap.TreeExplainer(model)

print("\nGenerating SHAP values...")
shap_values = explainer.shap_values(X)

if isinstance(shap_values, list):

    predicted_shap = np.zeros(
        (len(X), X.shape[1])
    )

    for i in range(len(X)):

        predicted_class = predictions[i]

        predicted_shap[i] = (
            shap_values[
                predicted_class
            ][i]
        )

else:

    predicted_shap = np.zeros(
        (len(X), X.shape[1])
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

shap_df = pd.DataFrame(
    predicted_shap,
    columns=X.columns
)

shap_df.to_csv(
    SHAP_DIR / "shap_train_values.csv",
    index=False
)

metadata_df = pd.DataFrame(
    {
        "True_Label":
            target_encoder.inverse_transform(
                y["Attack_type"]
            ),
        "Predicted_Label":
            target_encoder.inverse_transform(
                predictions
            ),
        "Confidence":
            confidence
    }
)

metadata_df.to_csv(
    SHAP_DIR / "shap_train_metadata.csv",
    index=False
)

print("\nSaved:")
print("shap_train_values.csv")
print("shap_train_metadata.csv")
