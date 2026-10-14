from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SHAP_DIR = PROJECT_ROOT / "data" / "shap"

SHAP_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_CLASS = 2000

print("=" * 70)
print("BUILDING TRAIN SHAP DATASET")
print("=" * 70)

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")

X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)

df = pd.concat([X_all, y_all], axis=1)

selected = []

for class_id in sorted(df["Attack_type"].unique()):

    class_df = df[df["Attack_type"] == class_id]

    count = len(class_df)

    if count <= TARGET_PER_CLASS:

        sampled = class_df

        print(
            f"Class {class_id:2d}: "
            f"{count:5d} (TAKE ALL)"
        )

    else:

        sampled = class_df.sample(
            n=TARGET_PER_CLASS,
            random_state=42
        )

        print(
            f"Class {class_id:2d}: "
            f"{TARGET_PER_CLASS:5d} (FROM {count})"
        )

    selected.append(sampled)

shap_df = pd.concat(
    selected,
    ignore_index=True
)

shap_df = shap_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_shap = shap_df.drop(
    columns=["Attack_type"]
)

y_shap = shap_df[
    ["Attack_type"]
]

X_shap.to_csv(
    SHAP_DIR / "X_train_shap.csv",
    index=False
)

y_shap.to_csv(
    SHAP_DIR / "y_train_shap.csv",
    index=False
)

print("\nSaved:")
print("X_train_shap.csv")
print("y_train_shap.csv")

print(f"\nTotal Samples: {len(shap_df):,}")
