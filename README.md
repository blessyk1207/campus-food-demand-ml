# Campus Food Demand Forecasting

An end-to-end machine learning project that predicts food demand using historical order data, pricing information, meal details, and promotional features.

## Problem Statement

Accurately forecasting food demand can help campus canteens and food-service providers plan inventory, reduce food waste, and prepare for expected orders.

This project explores historical food-order data and builds machine learning models to estimate the number of orders for a given meal and fulfilment center.

## Project Features

- Exploratory data analysis (EDA)
- Data preprocessing and feature engineering
- Time-based training and validation split
- Baseline models and machine learning model comparison
- Saved Random Forest model
- Reusable prediction script
- FastAPI prediction endpoint
- Automated API tests

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- FastAPI and Uvicorn
- Joblib
- Pytest

## Project Structure

```text
campus-food-demand-ml/
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── configs/
├── data/
│   └── raw/                 # Obtain dataset separately
├── models/
│   └── random_forest.joblib # Generated after training
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset

The project uses the Food Demand Forecasting dataset, which contains historical order quantities, meal identifiers, fulfilment center identifiers, prices, and promotional indicators.

The raw dataset is not included in this repository. Obtain the dataset from its original or an authorized source, review its license and usage terms, and place the required CSV files in `data/raw/`:

- `train.csv`
- `test_QoiMO9B.csv`
- `meal_info.csv`
- `fulfilment_center_info.csv`

## Installation

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/blessyk1207/campus-food-demand-ml
cd campus-food-demand-ml
```


### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Obtain the required CSV files from an authorized source, review the dataset license, and place them in `data/raw/`.


## Training and Evaluation

Run the training pipeline from the project root:

```bash
python -m src.models.train
```

The script trains Ridge Regression and Random Forest models, evaluates baseline approaches, and saves the Random Forest pipeline to `models/random_forest.joblib`.

## Current Model Results

Evaluation uses a time-based split: weeks 1–125 for training and weeks 126–145 for validation.

| Model | MAE | RMSE |
|---|---:|---:|
| Historical-average baseline | 107.07 | 243.68 |
| Random Forest with one-hot encoding | 126.37 | 231.32 |
| Recent-history baseline | 133.86 | 317.66 |
| Ridge Regression | 147.76 | 255.04 |
| Overall-mean baseline | 215.34 | 357.16 |

Lower values are better for both metrics.

The historical-average baseline achieves the lowest MAE, while Random Forest achieves the lowest RMSE among these experiments. Further model tuning and evaluation are needed before selecting a final production model.

## Making Predictions

After training and saving the model, run:

```bash
python -m src.models.predict
```

This loads the saved model and demonstrates a prediction for a sample input.

## Running the API

Start the FastAPI application:

```bash
python -m uvicorn api.main:app --reload
```

Open the interactive API documentation in your browser:

`http://127.0.0.1:8000/docs`

The API includes:

- `GET /` — basic service information
- `GET /health` — application and model status
- `POST /predict` — predict food demand from input features

Example prediction request:

```json
{
  "center_id": "55",
  "meal_id": "1885",
  "week_index": 144,
  "checkout_price": 150.0,
  "base_price": 160.0,
  "emailer_for_promotion": 0,
  "homepage_featured": 0,
  "discount_amount": 10.0,
  "discount_percentage": 6.25
}
```

The response contains a `predicted_orders` value. Predictions are estimates and are not guaranteed actual order quantities.

## Running Tests

Run the automated API tests:

```bash
python -m pytest tests/ -v
```

## Limitations and Future Improvements

- Improve model performance against the historical-average baseline.
- Evaluate errors across meals, fulfilment centers, and time periods.
- Add more robust input validation and prediction tests.
- Improve experiment tracking and model reproducibility.
- Containerize and deploy the API when needed.

## Disclaimer

This project is intended for educational and portfolio purposes. Forecasts should be evaluated further before being used for real operational decisions.
