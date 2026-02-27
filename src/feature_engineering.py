import pandas as pd


# ----------------------------------------
# Load Processed Weekly Store+Category Data
# ----------------------------------------

def load_processed_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ----------------------------------------
# Time Features
# ----------------------------------------

def create_time_features(df):

    df["year"] = df["Date"].dt.year
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Date"].dt.quarter

    return df


# ----------------------------------------
# Lag Features per Store + Category
# ----------------------------------------

def create_lag_features(df):

    df = df.sort_values(["StoreID", "Category", "Date"])

    df["lag_1"] = df.groupby(["StoreID", "Category"])["Revenue"].shift(1)
    df["lag_4"] = df.groupby(["StoreID", "Category"])["Revenue"].shift(4)
    df["lag_12"] = df.groupby(["StoreID", "Category"])["Revenue"].shift(12)

    return df


# ----------------------------------------
# Rolling Features per Store + Category
# ----------------------------------------

def create_rolling_features(df):

    df["rolling_mean_4"] = (
        df.groupby(["StoreID", "Category"])["Revenue"]
        .transform(lambda x: x.rolling(4).mean())
    )

    df["rolling_mean_12"] = (
        df.groupby(["StoreID", "Category"])["Revenue"]
        .transform(lambda x: x.rolling(12).mean())
    )

    return df


# ----------------------------------------
# Drop NA Rows
# ----------------------------------------

def drop_na_rows(df):
    return df.dropna()


# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":

    path = "data/processed/weekly_store_category_revenue.csv"

    df = load_processed_data(path)

    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    df = drop_na_rows(df)

    print("=" * 60)
    print("WEEKLY STORE + CATEGORY FEATURE ENGINEERED DATA")
    print("=" * 60)
    print(df.head())
    print("\nShape:", df.shape)
