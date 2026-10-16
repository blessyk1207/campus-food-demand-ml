
from pathlib import Path

import pandas as pd


def load_training_data() -> pd.DataFrame:
    """Load the original training dataset."""
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at: {data_path}"
        )

    df = pd.read_csv(data_path)

    required_columns = [
        "id",
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
        "num_orders",
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


if __name__ == "__main__":
    data = load_training_data()

    print("Training data loaded successfully!")
    print("Shape:", data.shape)
    print("Missing values:", data.isnull().sum().sum())
    print("Duplicate rows:", data.duplicated().sum())
    print("Columns:", data.columns.tolist())
