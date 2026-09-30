import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

DATA_PATH = Path("data/crop_disease_risk.csv")


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CROP DISEASE RISK DATASET VALIDATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)


# ---------------------------------------------------------
# DATA TYPES
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("DATA TYPES")
print("-" * 60)

print(df.dtypes)


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("MISSING VALUES")
print("-" * 60)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values are present.")


# ---------------------------------------------------------
# DUPLICATE ROWS
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("DUPLICATE ROWS")
print("-" * 60)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)


# ---------------------------------------------------------
# TARGET DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("TARGET DISTRIBUTION")
print("-" * 60)

print(df["risk_label"].value_counts())

print("\nTarget distribution percentage:")

print(
    df["risk_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ---------------------------------------------------------
# UNIQUE VALUES
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("UNIQUE VALUES")
print("-" * 60)

for column in df.columns:
    print(
        f"{column}: {df[column].nunique()} unique values"
    )


# ---------------------------------------------------------
# RISK SCORE ANALYSIS
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("RISK SCORE ANALYSIS")
print("-" * 60)

print(
    df.groupby("risk_label")["risk_score"]
    .agg(["min", "max", "mean"])
)


# ---------------------------------------------------------
# FINAL VALIDATION MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)