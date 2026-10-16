
from pathlib import Path

import joblib
import pandas as pd


def load_model():
    """Load the saved Random Forest pipeline."""
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "random_forest.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at: {model_path}. "
            "Run the training script first."
        )

    return joblib.load(model_path)


def predict_demand(input_data: pd.DataFrame):
    """Predict food demand for new records."""
    model = load_model()
    predictions = model.predict(input_data)

    results = input_data.copy()
    results["predicted_orders"] = predictions

    return results


if __name__ == "__main__":
    sample_data = pd.DataFrame([
        {
            "center_id": "55",
            "meal_id": "1885",
            "week_index": 145 - 1,
            "checkout_price": 150.0,
            "base_price": 160.0,
            "emailer_for_promotion": 0,
            "homepage_featured": 0,
            "discount_amount": 10.0,
            "discount_percentage": 6.25,
        }
    ])

    predictions = predict_demand(sample_data)

    print("\n--- Food Demand Prediction ---")
    print(predictions.to_string(index=False))
