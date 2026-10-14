from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

INPUT_DIM = X_test.shape[1]
NUM_CLASSES = len(target_encoder.classes_)

# ============================================================
# MODEL
# ============================================================

class TinyMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(INPUT_DIM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, x):
        return self.network(x)

model = TinyMLP().to(DEVICE)

model_path = (
    CHECKPOINT_DIR
    / "tinymlp_weights.pth"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=DEVICE
    )
)

model.eval()

# ============================================================
# PREDICTIONS
# ============================================================

X_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
).to(DEVICE)

start = time.perf_counter()

with torch.no_grad():

    logits = model(X_tensor)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    y_pred = torch.argmax(
        probabilities,
        dim=1
    ).cpu().numpy()

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

total_params = sum(
    p.numel()
    for p in model.parameters()
)

size_mb = (
    os.path.getsize(model_path)
    / (1024 * 1024)
)

print("\n" + "=" * 70)
print("MODEL INFO")
print("=" * 70)

print(f"Parameters        : {total_params:,}")
print(f"Model Size        : {size_mb:.4f} MB")

# ============================================================
# SINGLE SAMPLE
# ============================================================

sample_idx = 0

sample = torch.tensor(
    X_test.iloc[[sample_idx]].values,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    probs = torch.softmax(
        model(sample),
        dim=1
    )[0]

pred_class = torch.argmax(
    probs
).item()

confidence = probs[
    pred_class
].item()

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

top_probs, top_idx = torch.topk(
    probs,
    5
)

for prob, idx in zip(
    top_probs.cpu().numpy(),
    top_idx.cpu().numpy()
):

    label = target_encoder.inverse_transform(
        [idx]
    )[0]

    print(
        f"{label:<25} {prob:.4f}"
    )
