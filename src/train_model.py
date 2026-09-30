"""
train_model.py
================================================================
Crop Disease Risk Classification — Standalone Training Script
----------------------------------------------------------------
Loads the dataset, engineers the growth-stage feature, builds a
preprocessing + classification pipeline, trains and compares three
foundational models (Logistic Regression, Decision Tree, KNN),
selects the best model by macro F1-score, and saves the final
trained pipeline to disk for use by app.py / predict.py.

Run from the project root:
    python train_model.py
================================================================
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/crop_disease_risk.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "crop_disease_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


# ============================================================
# STEP 2 — FEATURE ENGINEERING
# ============================================================

def assign_growth_stage(day: int) -> str:
    if day <= 90:
        return "Seedling"
    elif day <= 180:
        return "Vegetative"
    elif day <= 270:
        return "Flowering"
    else:
        return "Maturity"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["growth_stage"] = df["day_of_year"].apply(assign_growth_stage)
    return df


# ============================================================
# STEP 3 — PREPROCESSING PIPELINE
# ============================================================

CATEGORICAL_FEATURES = ["crop", "disease", "growth_stage"]

NUMERICAL_FEATURES = [
    "day_of_year",
    "avg_temp_c",
    "avg_humidity_pct",
    "rainfall_mm",
    "leaf_wetness_hours",
    "wind_speed_kmph",
    "soil_moisture_pct",
    "prior_regional_incidence_pct",
]


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(transformers=[
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
    ])


# ============================================================
# STEP 4 — MODEL DEFINITIONS
# ============================================================

def build_candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


# ============================================================
# STEP 5 — TRAIN, COMPARE, SELECT BEST MODEL
# ============================================================

def train_and_compare(X_train, X_test, y_train, y_test, preprocessor):
    results = []
    fitted_pipelines = {}

    for name, estimator in build_candidate_models().items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        print(f"{name} training completed.")

        y_pred = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })

        fitted_pipelines[name] = pipeline

        print(f"\n{'=' * 60}\n{name} — Classification Report\n{'=' * 60}")
        print(classification_report(y_test, y_pred, digits=4))

    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best_model_name = results_df.loc[0, "Model"]
    print(f"\nBest model based on Macro F1-score: {best_model_name}")

    return fitted_pipelines[best_model_name], best_model_name, results_df


# ============================================================
# STEP 6 — SAVE MODEL
# ============================================================

def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModel saved successfully. Location: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_data()
    df = engineer_features(df)

    drop_columns = ["risk_label", "risk_score"]  # risk_score excluded to avoid target leakage
    X = df.drop(columns=drop_columns)
    y = df["risk_label"]

    print(f"\nFeature shape: {X.shape}")
    print(f"Target classes: {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()

    best_model, best_model_name, results_df = train_and_compare(
        X_train, X_test, y_train, y_test, preprocessor
    )

    save_model(best_model, MODEL_PATH)

    print("\n" + "=" * 50)
    print("Project Model Information")
    print("=" * 50)
    print(f"Selected Model: {best_model_name}")
    print(f"Model File: {MODEL_PATH}")
    print(f"Number of Training Rows: {len(X_train)}")
    print(f"Number of Original Features: {X.shape[1]}")
    print("\nFeatures used by model:")
    for feature in X.columns:
        print(f"- {feature}")
    print("\nExcluded from model:")
    print("- risk_label  -> target")
    print("- risk_score  -> excluded to avoid target leakage")


if __name__ == "__main__":
    main()