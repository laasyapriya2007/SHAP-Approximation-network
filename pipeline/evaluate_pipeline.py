from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "shap"
)

print("=" * 70)
print("PIPELINE EVALUATION")
print("=" * 70)

#
# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
#

results = pd.read_csv(
    SHAP_DIR
    / "pipeline_results.csv"
)

exact_shap = pd.read_csv(
    SHAP_DIR
    / "pipeline_exact_shap.csv"
)

approx_shap = pd.read_csv(
    SHAP_DIR
    / "pipeline_approx_shap.csv"
)

print(f"\nSamples: {len(results):,}")

#
# --------------------------------------------------
# IDS EVALUATION
# --------------------------------------------------
#

print("\n" + "=" * 70)
print("IDS EVALUATION")
print("=" * 70)

y_true = results["True_Label"]
y_pred = results["Predicted_Label"]

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    digits=4,
    zero_division=0
)

print("\nClassification Report\n")
print(report)

with open(
    SHAP_DIR / "pipeline_classification_report.txt",
    "w"
) as f:

    f.write(report)

labels = sorted(
    list(
        set(y_true)
        |
        set(y_pred)
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    SHAP_DIR
    / "pipeline_ids_confusion_matrix.csv"
)

print("\nConfusion Matrix")

print(cm_df)

#
# --------------------------------------------------
# SHAP APPROXIMATOR EVALUATION
# --------------------------------------------------
#

print("\n" + "=" * 70)
print("SHAP APPROXIMATOR EVALUATION")
print("=" * 70)

y_true_shap = exact_shap.values
y_pred_shap = approx_shap.values

mse = mean_squared_error(
    y_true_shap,
    y_pred_shap
)

rmse = np.sqrt(
    mse
)

mae = mean_absolute_error(
    y_true_shap,
    y_pred_shap
)

r2 = r2_score(
    y_true_shap,
    y_pred_shap
)

dot = np.sum(
    y_true_shap
    *
    y_pred_shap,
    axis=1
)

norm_true = np.linalg.norm(
    y_true_shap,
    axis=1
)

norm_pred = np.linalg.norm(
    y_pred_shap,
    axis=1
)

cosine_scores = (

    dot

    /

    (
        norm_true
        *
        norm_pred
        +
        1e-10
    )
)

avg_cosine = np.mean(
    cosine_scores
)

print(f"\nMSE               : {mse:.8f}")
print(f"RMSE              : {rmse:.8f}")
print(f"MAE               : {mae:.8f}")
print(f"R² Score          : {r2:.8f}")
print(f"Cosine Similarity : {avg_cosine:.8f}")

shap_metrics = pd.DataFrame({

    "Metric": [
        "MSE",
        "RMSE",
        "MAE",
        "R2",
        "Cosine Similarity"
    ],

    "Value": [
        mse,
        rmse,
        mae,
        r2,
        avg_cosine
    ]
})

shap_metrics.to_csv(
    SHAP_DIR
    / "pipeline_shap_metrics.csv",
    index=False
)

#
# --------------------------------------------------
# AUTOENCODER EVALUATION
# --------------------------------------------------
#

print("\n" + "=" * 70)
print("AUTOENCODER EVALUATION")
print("=" * 70)

errors = results[
    "Reconstruction_Error"
]

print(f"\nMean Error   : {errors.mean():.6f}")
print(f"Median Error : {errors.median():.6f}")
print(f"Std Error    : {errors.std():.6f}")
print(f"Min Error    : {errors.min():.6f}")
print(f"Max Error    : {errors.max():.6f}")

normal_errors = results[
    results["True_Label"]
    ==
    "Normal"
]["Reconstruction_Error"]

attack_errors = results[
    results["True_Label"]
    !=
    "Normal"
]["Reconstruction_Error"]

print("\nNormal vs Attack")

print(
    f"\nNormal Mean Error : "
    f"{normal_errors.mean():.6f}"
)

print(
    f"Attack Mean Error : "
    f"{attack_errors.mean():.6f}"
)

if normal_errors.mean() > 0:

    print(
        f"Separation Ratio  : "
        f"{attack_errors.mean() / normal_errors.mean():.2f}x"
    )

class_stats = (

    results

    .groupby(
        "True_Label"
    )[
        "Reconstruction_Error"
    ]

    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )

    .sort_values(
        by="mean",
        ascending=False
    )
)

print("\nClass-wise Reconstruction Errors\n")

print(class_stats)

class_stats.to_csv(
    SHAP_DIR
    / "pipeline_class_error_stats.csv"
)

#
# --------------------------------------------------
# TIMING ANALYSIS
# --------------------------------------------------
#

print("\n" + "=" * 70)
print("TIMING ANALYSIS")
print("=" * 70)

timing_columns = [

    "IDS_Time",

    "SHAP_Time",

    "Autoencoder_Time",

    "Total_Pipeline_Time"
]

for col in timing_columns:

    print(f"\n{col}")

    print(
        f"Mean   : "
        f"{results[col].mean():.6f}"
    )

    print(
        f"Median : "
        f"{results[col].median():.6f}"
    )

    print(
        f"Min    : "
        f"{results[col].min():.6f}"
    )

    print(
        f"Max    : "
        f"{results[col].max():.6f}"
    )

    print(
        f"Std    : "
        f"{results[col].std():.6f}"
    )

#
# --------------------------------------------------
# SAVE OVERALL SUMMARY
# --------------------------------------------------
#

summary = pd.DataFrame({

    "Metric": [

        "IDS Accuracy",
        "IDS Precision",
        "IDS Recall",
        "IDS F1",

        "SHAP MSE",
        "SHAP RMSE",
        "SHAP MAE",
        "SHAP R2",
        "SHAP Cosine",

        "AE Mean Error",
        "AE Median Error",

        "IDS Mean Time",
        "SHAP Mean Time",
        "AE Mean Time",
        "Pipeline Mean Time"
    ],

    "Value": [

        accuracy,
        precision,
        recall,
        f1,

        mse,
        rmse,
        mae,
        r2,
        avg_cosine,

        errors.mean(),
        errors.median(),

        results["IDS_Time"].mean(),
        results["SHAP_Time"].mean(),
        results["Autoencoder_Time"].mean(),
        results["Total_Pipeline_Time"].mean()
    ]
})

summary.to_csv(
    SHAP_DIR
    / "pipeline_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    SHAP_DIR / "pipeline_summary.csv"
)

print(
    SHAP_DIR / "pipeline_ids_confusion_matrix.csv"
)

print(
    SHAP_DIR / "pipeline_shap_metrics.csv"
)

print(
    SHAP_DIR / "pipeline_class_error_stats.csv"
)

print("\n" + "=" * 70)
print("PIPELINE EVALUATION COMPLETED")
print("=" * 70)
