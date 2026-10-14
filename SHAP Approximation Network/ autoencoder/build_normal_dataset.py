from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "shap"
)

print("=" * 70)
print("BUILDING NORMAL SHAP DATASETS")
print("=" * 70)

train_shap = pd.read_csv(
    SHAP_DIR / "shap_train_values.csv"
)

train_meta = pd.read_csv(
    SHAP_DIR / "shap_train_metadata.csv"
)

test_shap = pd.read_csv(
    SHAP_DIR / "shap_test_values.csv"
)

test_meta = pd.read_csv(
    SHAP_DIR / "shap_test_metadata.csv"
)

print("\nExtracting normal training samples...")

normal_train_idx = (
    train_meta["True_Label"]
    == "Normal"
)

normal_train = train_shap[
    normal_train_idx
]

print(
    f"Normal Train Samples: "
    f"{len(normal_train)}"
)

print("\nExtracting normal test samples...")

normal_test_idx = (
    test_meta["True_Label"]
    == "Normal"
)

normal_test = test_shap[
    normal_test_idx
]

print(
    f"Normal Test Samples: "
    f"{len(normal_test)}"
)

normal_train.to_csv(
    SHAP_DIR
    / "normal_shap_train_values.csv",
    index=False
)

normal_test.to_csv(
    SHAP_DIR
    / "normal_shap_test_values.csv",
    index=False
)

print("\nSaved:")

print(
    SHAP_DIR
    / "normal_shap_train_values.csv"
)

print(
    SHAP_DIR
    / "normal_shap_test_values.csv"
)
