import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


# ----------------------------------------
# Load Data
# ----------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ----------------------------------------
# Feature Engineering
# ----------------------------------------

from feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    drop_na_rows
)


def prepare_features(df):
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = drop_na_rows(df)
    return df


# ----------------------------------------
# Time Split
# ----------------------------------------

def time_split(df):
    df = df.sort_values("Date")
    split_index = int(len(df) * 0.8)
    return df.iloc[:split_index], df.iloc[split_index:]


# ----------------------------------------
# SMAPE
# ----------------------------------------

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred))
    )


# ----------------------------------------
# Train Per Store + Category
# ----------------------------------------

def train_models(df):

    groups = df.groupby(["StoreID", "Category"])

    features = [
        "year",
        "week_of_year",
        "quarter",
        "lag_1",
        "lag_4",
        "lag_12",
        "rolling_mean_4",
        "rolling_mean_12",
    ]

    for (store, category), group_df in groups:

        if len(group_df) < 30:
            continue

        print("\n" + "=" * 70)
        print(f"STORE: {store} | CATEGORY: {category}")
        print("=" * 70)

        train, test = time_split(group_df)

        X_train = train[features]
        y_train = train["Revenue"]

        X_test = test[features]
        y_test = test["Revenue"]

        # Baseline
        baseline_pred = test["lag_1"]
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_smape = smape(y_test, baseline_pred)

        # Random Forest
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        rf_pred = model.predict(X_test)

        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_smape = smape(y_test, rf_pred)

        improvement = ((baseline_mae - rf_mae) / baseline_mae) * 100

        print(f"Baseline MAE : {baseline_mae:.2f}")
        print(f"RF MAE       : {rf_mae:.2f}")
        print(f"MAE Improvement: {improvement:.2f}%")
        print(f"RF SMAPE     : {rf_smape:.2f}%")

        # Save model
        model_name = f"models/weekly_model_{store}_{category}.pkl"
        joblib.dump(model, model_name)


# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":

    path = "data/processed/weekly_store_category_revenue.csv"

    df = load_data(path)
    df = prepare_features(df)

    train_models(df)
