from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "shap"
)

print("=" * 70)
print("RECONSTRUCTION ERROR ANALYSIS")
print("=" * 70)

results = pd.read_csv(
    SHAP_DIR
    / "reconstruction_errors.csv"
)

print(f"\nTotal Samples : {len(results):,}")

print("\nClasses Found:")

for cls in sorted(
    results["True_Label"].unique()
):
    print(cls)

print("\n" + "=" * 70)
print("CLASS-WISE STATISTICS")
print("=" * 70)

stats = (

    results

    .groupby("True_Label")[
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

print(stats)

stats.to_csv(
    SHAP_DIR
    / "class_reconstruction_statistics.csv"
)

print("\nSaved:")
print(
    SHAP_DIR
    / "class_reconstruction_statistics.csv"
)

print("\n" + "=" * 70)
print("TOP 10 HIGHEST ERRORS")
print("=" * 70)

highest = (

    results

    .sort_values(
        by="Reconstruction_Error",
        ascending=False
    )

    .head(10)
)

print(highest)

highest.to_csv(
    SHAP_DIR
    / "top10_highest_errors.csv",
    index=False
)

print("\n" + "=" * 70)
print("TOP 10 LOWEST ERRORS")
print("=" * 70)

lowest = (

    results

    .sort_values(
        by="Reconstruction_Error",
        ascending=True
    )

    .head(10)
)

print(lowest)

lowest.to_csv(
    SHAP_DIR
    / "top10_lowest_errors.csv",
    index=False
)

print("\n" + "=" * 70)
print("15 RANDOM SAMPLES PER CLASS")
print("=" * 70)

sampled_rows = []

for cls in sorted(
    results["True_Label"].unique()
):

    class_df = results[
        results["True_Label"] == cls
    ]

    n_samples = min(
        15,
        len(class_df)
    )

    sampled = class_df.sample(
        n=n_samples,
        random_state=42
    )

    sampled_rows.append(
        sampled
    )

sample_analysis = pd.concat(
    sampled_rows,
    ignore_index=True
)

sample_analysis = sample_analysis.sort_values(
    by="True_Label"
)

print(sample_analysis)

sample_analysis.to_csv(
    SHAP_DIR
    / "sample_reconstruction_analysis.csv",
    index=False
)

print("\nSaved:")
print(
    SHAP_DIR
    / "sample_reconstruction_analysis.csv"
)

print("\n" + "=" * 70)
print("MEAN ERROR RANKING")
print("=" * 70)

ranking = (

    results

    .groupby(
        "True_Label"
    )[
        "Reconstruction_Error"
    ]

    .mean()

    .sort_values(
        ascending=False
    )
)

for i, (cls, value) in enumerate(
    ranking.items(),
    start=1
):

    print(
        f"{i:2d}. "
        f"{cls:<25} "
        f"{value:.6f}"
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)
