
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from the raw food-demand data."""
    data = df.copy()

    # Convert week into a numeric feature relative to the first week.
    data["week_index"] = data["week"] - data["week"].min()

    # Price difference between base price and checkout price.
    data["discount_amount"] = (
        data["base_price"] - data["checkout_price"]
    )

    # Discount percentage, safely handling zero base prices.
    data["discount_percentage"] = (
        data["discount_amount"]
        / data["base_price"].replace(0, float("nan"))
    ) * 100

    # Center and meal IDs are categorical identifiers, not quantities.
    data["center_id"] = data["center_id"].astype(str)
    data["meal_id"] = data["meal_id"].astype(str)

    return data


if __name__ == "__main__":
    from src.data.preprocessing import load_training_data

    df = load_training_data()
    featured_df = engineer_features(df)

    print("Feature engineering successful!")
    print("Shape:", featured_df.shape)
    print("New columns:", [
        "week_index",
        "discount_amount",
        "discount_percentage",
    ])
    print(featured_df.head())
