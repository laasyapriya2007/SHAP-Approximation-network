from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SHAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "shap"
)

SHAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_PER_CLASS = 2000

print("=" * 70)
print("BUILDING SHAP DATASET")
print("=" * 70)

print("\nLoading processed datasets...")

X_train = pd.read_csv(
    PROCESSED_DIR / "X_train.csv"
)

X_test = pd.read_csv(
    PROCESSED_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    PROCESSED_DIR / "y_train.csv"
)

y_test = pd.read_csv(
    PROCESSED_DIR / "y_test.csv"
)

print("Combining train and test datasets...")

X_all = pd.concat(
    [X_train, X_test],
    ignore_index=True
)

y_all = pd.concat(
    [y_train, y_test],
    ignore_index=True
)

df = pd.concat(
    [X_all, y_all],
    axis=1
)

print(f"\nCombined Shape: {df.shape}")

print("\nClass Distribution Before Sampling")

class_counts = (
    df["Attack_type"]
    .value_counts()
    .sort_index()
)

print(class_counts)

selected_dfs = []

print("\nSampling per class...\n")

for class_id in sorted(df["Attack_type"].unique()):

    class_df = df[
        df["Attack_type"] == class_id
    ]

    count = len(class_df)

    if count <= TARGET_PER_CLASS:

        sampled_df = class_df

        print(
            f"Class {class_id:2d} : "
            f"{count:5d} samples "
            f"(TAKE ALL)"
        )

    else:

        sampled_df = class_df.sample(
            n=TARGET_PER_CLASS,
            random_state=42
        )

        print(
            f"Class {class_id:2d} : "
            f"{TARGET_PER_CLASS:5d} samples "
            f"(FROM {count})"
        )

    selected_dfs.append(
        sampled_df
    )

print("\nCombining sampled classes...")

shap_df = pd.concat(
    selected_dfs,
    ignore_index=True
)

shap_df = shap_df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)

print(
    f"\nFinal SHAP Dataset Shape: "
    f"{shap_df.shape}"
)

print("\nFinal Class Distribution")

print(
    shap_df["Attack_type"]
    .value_counts()
    .sort_index()
)

X_shap = shap_df.drop(
    columns=["Attack_type"]
)

y_shap = shap_df[
    ["Attack_type"]
]

print("\nSaving files...")

X_shap.to_csv(
    SHAP_DIR / "X_shap.csv",
    index=False
)

y_shap.to_csv(
    SHAP_DIR / "y_shap.csv",
    index=False
)

print("\nSaved:")

print(
    SHAP_DIR / "X_shap.csv"
)

print(
    SHAP_DIR / "y_shap.csv"
)

print("\n" + "=" * 70)
print("SHAP DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nTotal Samples : "
    f"{len(shap_df):,}"
)

print(
    f"Features      : "
    f"{X_shap.shape[1]}"
)

print(
    f"Classes       : "
    f"{y_shap['Attack_type'].nunique()}"
)
