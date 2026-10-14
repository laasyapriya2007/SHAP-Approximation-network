from pathlib import Path
import pandas as pd
import joblib

from lightgbm import LGBMClassifier

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

print("\nCreating LightGBM model...")

model = LGBMClassifier(
    objective="multiclass",
    num_class=15,

    n_estimators=100,
    learning_rate=0.01,

    max_depth=8,
    num_leaves=31,

    random_state=42,
    n_jobs=-1,

    verbose=-1
)

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed")

model_path = CHECKPOINT_DIR / "lightgbm_weights.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved to:\n{model_path}")

print("\nDone!")
