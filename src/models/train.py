
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocessing import load_training_data
from src.features.engineering import engineer_features


def main():
    # Load data and create features
    df = load_training_data()
    df = engineer_features(df)

    # Keep the final 20 weeks for validation
    validation_start_week = df["week"].max() - 19

    train_df = df[df["week"] < validation_start_week].copy()
    valid_df = df[df["week"] >= validation_start_week].copy()

    target = "num_orders"

    categorical_features = ["center_id", "meal_id"]

    numerical_features = [
        "week_index",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
        "discount_amount",
        "discount_percentage",
    ]

    features = categorical_features + numerical_features

    X_train = train_df[features]
    y_train = train_df[target]

    X_valid = valid_df[features]
    y_valid = valid_df[target]

    # Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # Model 1: Ridge Regression
    ridge_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0)),
    ])

    ridge_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_valid)

    ridge_mae = mean_absolute_error(y_valid, ridge_predictions)
    ridge_rmse = mean_squared_error(y_valid, ridge_predictions) ** 0.5

    print("\n--- Ridge Regression ---")
    print(f"Ridge MAE:  {ridge_mae:.2f}")
    print(f"Ridge RMSE: {ridge_rmse:.2f}")

    print("\n--- Dataset Split ---")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(valid_df))
    print(
        "Training weeks:",
        train_df["week"].min(),
        "to",
        train_df["week"].max(),
    )
    print(
        "Validation weeks:",
        valid_df["week"].min(),
        "to",
        valid_df["week"].max(),
    )

    # Baseline 1: historical average for each center-meal pair
    historical_averages = (
        train_df.groupby(["center_id", "meal_id"])["num_orders"]
        .mean()
        .rename("baseline_prediction")
        .reset_index()
    )

    baseline_df = valid_df[["center_id", "meal_id"]].merge(
        historical_averages,
        on=["center_id", "meal_id"],
        how="left",
    )

    baseline_predictions = baseline_df["baseline_prediction"].fillna(
        train_df["num_orders"].mean()
    ).to_numpy()

    baseline_mae = mean_absolute_error(y_valid, baseline_predictions)
    baseline_rmse = mean_squared_error(
        y_valid, baseline_predictions
    ) ** 0.5

    print("\n--- Historical-Average Baseline ---")
    print(f"Baseline MAE:  {baseline_mae:.2f}")
    print(f"Baseline RMSE: {baseline_rmse:.2f}")

    # Baseline 2: overall training-set mean
    overall_mean_predictions = [
        train_df["num_orders"].mean()
    ] * len(valid_df)

    overall_mean_mae = mean_absolute_error(
        y_valid, overall_mean_predictions
    )
    overall_mean_rmse = mean_squared_error(
        y_valid, overall_mean_predictions
    ) ** 0.5

    print("\n--- Overall-Mean Baseline ---")
    print(f"Overall-mean MAE:  {overall_mean_mae:.2f}")
    print(f"Overall-mean RMSE: {overall_mean_rmse:.2f}")

    # Model 2: Random Forest with proper categorical encoding
    rf_preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    rf_model = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_valid)

    rf_mae = mean_absolute_error(y_valid, rf_predictions)
    rf_rmse = mean_squared_error(y_valid, rf_predictions) ** 0.5

    print("\n--- Random Forest with One-Hot Encoding ---")
    print(f"Random Forest MAE:  {rf_mae:.2f}")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")
     # Save the trained Random Forest pipeline
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "random_forest.joblib"
    joblib.dump(rf_model, model_path)

    print(f"Saved Random Forest model to: {model_path}")
    # Baseline 3: average demand over the last 4 training weeks
    recent_weeks = sorted(train_df["week"].unique())[-4:]

    recent_averages = (
        train_df[train_df["week"].isin(recent_weeks)]
        .groupby(["center_id", "meal_id"])["num_orders"]
        .mean()
        .rename("recent_prediction")
        .reset_index()
    )

    recent_df = valid_df[["center_id", "meal_id"]].merge(
        recent_averages,
        on=["center_id", "meal_id"],
        how="left",
    )

    recent_predictions = recent_df["recent_prediction"].fillna(
        train_df["num_orders"].mean()
    ).to_numpy()

    recent_mae = mean_absolute_error(y_valid, recent_predictions)
    recent_rmse = mean_squared_error(
        y_valid, recent_predictions
    ) ** 0.5

    print("\n--- Recent-History Baseline ---")
    print("Weeks used:", recent_weeks)
    print(f"Recent baseline MAE:  {recent_mae:.2f}")
    print(f"Recent baseline RMSE: {recent_rmse:.2f}")


if __name__ == "__main__":
    main()
