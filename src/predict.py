"""
predict.py
================================================================
Crop Disease Risk Classification — Standalone CLI Prediction
----------------------------------------------------------------
Loads the trained model and returns a risk prediction (Low /
Medium / High) with full class probabilities for a single
observation, either using the built-in sample values or values
supplied via command-line arguments.

Examples
--------
Run with default sample values:
    python predict.py

Run with custom values:
    python predict.py --crop Rice --disease Blast --growth_stage Vegetative \\
        --day_of_year 180 --avg_temp_c 28 --avg_humidity_pct 82 \\
        --rainfall_mm 120 --leaf_wetness_hours 10 --wind_speed_kmph 6 \\
        --soil_moisture_pct 65 --prior_regional_incidence_pct 35
================================================================
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/crop_disease_model.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict crop disease risk from crop, disease, and environmental readings."
    )

    parser.add_argument("--crop", default="Rice",
                         choices=["Cotton", "Rice", "Potato", "Wheat", "Grape"])
    parser.add_argument("--disease", default="Blast",
                         choices=["Bacterial Blight", "Blast", "Late Blight", "Stripe Rust", "Powdery Mildew"])
    parser.add_argument("--growth_stage", default="Vegetative",
                         choices=["Seedling", "Vegetative", "Flowering", "Maturity"])
    parser.add_argument("--day_of_year", type=int, default=180)
    parser.add_argument("--avg_temp_c", type=float, default=28.0)
    parser.add_argument("--avg_humidity_pct", type=float, default=82.0)
    parser.add_argument("--rainfall_mm", type=float, default=120.0)
    parser.add_argument("--leaf_wetness_hours", type=float, default=10.0)
    parser.add_argument("--wind_speed_kmph", type=float, default=6.0)
    parser.add_argument("--soil_moisture_pct", type=float, default=65.0)
    parser.add_argument("--prior_regional_incidence_pct", type=float, default=35.0)

    return parser.parse_args()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            f"Run 'python train_model.py' first to generate it."
        )
    return joblib.load(MODEL_PATH)


def main():
    args = parse_args()
    model = load_model()

    input_data = pd.DataFrame({
        "crop": [args.crop],
        "disease": [args.disease],
        "growth_stage": [args.growth_stage],
        "day_of_year": [args.day_of_year],
        "avg_temp_c": [args.avg_temp_c],
        "avg_humidity_pct": [args.avg_humidity_pct],
        "rainfall_mm": [args.rainfall_mm],
        "leaf_wetness_hours": [args.leaf_wetness_hours],
        "wind_speed_kmph": [args.wind_speed_kmph],
        "soil_moisture_pct": [args.soil_moisture_pct],
        "prior_regional_incidence_pct": [args.prior_regional_incidence_pct],
    })

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    print("\n" + "=" * 50)
    print("CROP DISEASE RISK PREDICTION")
    print("=" * 50)

    print("\nInput:")
    for col in input_data.columns:
        print(f"- {col}: {input_data[col].iloc[0]}")

    print(f"\nPredicted Disease Risk: {prediction}")

    print("\nClass Probabilities:")
    prob_df = pd.DataFrame({
        "Risk Category": classes,
        "Probability (%)": (probabilities * 100).round(2),
    }).sort_values(by="Probability (%)", ascending=False)

    print(prob_df.to_string(index=False))


if __name__ == "__main__":
    main()