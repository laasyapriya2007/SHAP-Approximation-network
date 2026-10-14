from pathlib import Path
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
).values

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).values.ravel()

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

INPUT_DIM = X_train.shape[1]
NUM_CLASSES = len(torch.unique(y_train))

# ============================================================
# DATASET
# ============================================================

dataset = TensorDataset(
    X_train,
    y_train
)

val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False
)

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

# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train.numpy()),
    y=y_train.numpy()
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

# ============================================================
# TRAINING
# ============================================================

EPOCHS = 100

best_f1 = 0
best_weights = None

early_stop_patience = 10
counter = 0

print("\nTraining started...\n")

for epoch in range(EPOCHS):

    # --------------------
    # TRAIN
    # --------------------

    model.train()

    train_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    # --------------------
    # VALIDATION
    # --------------------

    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(DEVICE)

            outputs = model(batch_x)

            preds = torch.argmax(
                outputs,
                dim=1
            ).cpu().numpy()

            val_preds.extend(preds)
            val_targets.extend(batch_y.numpy())

    val_f1 = f1_score(
        val_targets,
        val_preds,
        average="weighted"
    )

    scheduler.step(val_f1)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss={train_loss/len(train_loader):.4f} "
        f"Val_F1={val_f1:.4f}"
    )

    # --------------------
    # SAVE BEST MODEL
    # --------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        counter = 0

    else:

        counter += 1

    if counter >= early_stop_patience:

        print("\nEarly stopping triggered")
        break

# ============================================================
# SAVE BEST MODEL
# ============================================================

torch.save(
    best_weights,
    CHECKPOINT_DIR / "tinymlp_weights.pth"
)

print(f"\nBest Validation F1 : {best_f1:.4f}")

print("\nModel saved successfully")
