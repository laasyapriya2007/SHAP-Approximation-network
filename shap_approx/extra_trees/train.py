from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_DIR = PROJECT_ROOT / "data" / "shap"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 70)
print("TRAINING EXTRA TREES SHAP APPROXIMATOR")
print("=" * 70)

X_train = pd.read_csv(
    SHAP_DIR / "X_train_shap.csv"
)

y_train = pd.read_csv(
    SHAP_DIR / "shap_train_values.csv"
)

print(f"\nX Shape : {X_train.shape}")
print(f"Y Shape : {y_train.shape}")

base_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
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
    / "shap_extra_trees_weights.pkl"
)

print("\nModel Saved")

print(
    CHECKPOINT_DIR
    / "shap_extra_trees_weights.pkl"
)
