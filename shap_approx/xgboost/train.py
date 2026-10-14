from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_DIR = PROJECT_ROOT / "data" / "shap"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 70)
print("TRAINING XGBOOST SHAP APPROXIMATOR")
print("=" * 70)

X_train = pd.read_csv(
    SHAP_DIR / "X_train_shap.csv"
)

y_train = pd.read_csv(
    SHAP_DIR / "shap_train_values.csv"
)

print(f"\nX Shape : {X_train.shape}")
print(f"Y Shape : {y_train.shape}")

base_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

model = MultiOutputRegressor(
    base_model,
    n_jobs=-1
)

print("\nTraining...\n")

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    CHECKPOINT_DIR
    / "shap_xgboost_weights.pkl"
)

print("\nModel Saved")

print(
    CHECKPOINT_DIR
    / "shap_xgboost_weights.pkl"
)
