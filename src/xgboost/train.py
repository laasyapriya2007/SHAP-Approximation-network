from pathlib import Path
import pandas as pd
import joblib

from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")

print("\nCreating XGBoost model...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=15,
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed")

model_path = CHECKPOINT_DIR / "xgboost_weights.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved to:\n{model_path}")

print("\nDone!")
