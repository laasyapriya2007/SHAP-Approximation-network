from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import time
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

print("=" * 70)
print("EVALUATING XGBOOST IDS")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).values.ravel()

target_encoder = joblib.load(
    DATA_DIR / "target_encoder.pkl"
)

model_path = (
    CHECKPOINT_DIR
    / "xgboost_weights.pkl"
)

model = joblib.load(
    model_path
)

print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")

# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")

start = time.perf_counter()

y_pred = model.predict(
    X_test
)

end = time.perf_counter()

inference_time = (
    end - start
)

per_sample_time = (
    inference_time
    / len(X_test)
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 70)
print("METRICS")
print("=" * 70)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"Inference Time    : {inference_time:.6f} sec")
print(f"Per Sample Time   : {per_sample_time * 1e6:.2f} us")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ============================================================
# MODEL INFO
# ============================================================

size_mb = (
    os.path.getsize(model_path)
    / (1024 * 1024)
)

print("\n" + "=" * 70)
print("MODEL INFO")
print("=" * 70)

print(f"Model Size        : {size_mb:.4f} MB")

# ============================================================
# SINGLE SAMPLE
# ============================================================

sample_idx = 0

sample = X_test.iloc[
    [sample_idx]
]

pred_class = model.predict(
    sample
)[0]

probabilities = model.predict_proba(
    sample
)[0]

confidence = np.max(
    probabilities
)

predicted_label = target_encoder.inverse_transform(
    [pred_class]
)[0]

actual_label = target_encoder.inverse_transform(
    [y_test[sample_idx]]
)[0]

print("\n" + "=" * 70)
print("SINGLE SAMPLE")
print("=" * 70)

print(f"Actual Class      : {actual_label}")
print(f"Predicted Class   : {predicted_label}")
print(f"Confidence        : {confidence:.4f}")

print("\nTop 5 Probabilities")

top_indices = np.argsort(
    probabilities
)[::-1][:5]

for idx in top_indices:

    class_name = target_encoder.inverse_transform(
        [idx]
    )[0]

    print(
        f"{class_name:<25} "
        f"{probabilities[idx]:.4f}"
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURE IMPORTANCES")
print("=" * 70)

feature_importance = pd.DataFrame(
    {
        "Feature": X_test.columns,
        "Importance": model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.head(20)
)
