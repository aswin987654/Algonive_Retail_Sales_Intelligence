import pandas as pd


# --------------------------------------------------
# LOAD EXCEL SHEETS
# --------------------------------------------------

def load_sheets(path):
    customers = pd.read_excel(path, sheet_name="Customers")
    products = pd.read_excel(path, sheet_name="Products")
    stores = pd.read_excel(path, sheet_name="Stores")
    transactions = pd.read_excel(path, sheet_name="Transactions")

    return customers, products, stores, transactions


# --------------------------------------------------
# MERGE RELATIONAL TABLES
# --------------------------------------------------

def merge_data(customers, products, stores, transactions):

    # Merge transactions with products
    df = transactions.merge(products, on="ProductID", how="left")

    # Merge with stores
    df = df.merge(stores, on="StoreID", how="left")

    # Merge with customers
    df = df.merge(customers, on="CustomerID", how="left")

    return df


# --------------------------------------------------
# CALCULATE BUSINESS METRICS
# --------------------------------------------------

def calculate_revenue_profit(df):

    df["Revenue"] = (
        df["Quantity"] * df["UnitPrice"] * (1 - df["Discount"])
    )

    df["Profit"] = (
        (df["UnitPrice"] - df["CostPrice"])
        * df["Quantity"]
        * (1 - df["Discount"])
    )

    return df


# --------------------------------------------------
# AGGREGATE WEEKLY STORE + CATEGORY REVENUE
# --------------------------------------------------

def aggregate_weekly_store_category_revenue(df):

    df["Date"] = pd.to_datetime(df["Date"])

    weekly_store_category = (
        df
        .groupby([
            pd.Grouper(key="Date", freq="W"),
            "StoreID",
            "Category"
        ])["Revenue"]
        .sum()
        .reset_index()
    )

    weekly_store_category = weekly_store_category.sort_values(
        ["StoreID", "Category", "Date"]
    )

    return weekly_store_category


# --------------------------------------------------
# SAVE PROCESSED DATA
# --------------------------------------------------

def save_processed_data(df, path):
    df.to_csv(path, index=False)


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    path = "data/raw/retail_sales_dataset.xlsx"

    # Load
    customers, products, stores, transactions = load_sheets(path)

    # Merge
    merged_df = merge_data(customers, products, stores, transactions)

    # Calculate revenue & profit
    final_df = calculate_revenue_profit(merged_df)

    # Aggregate weekly by Store + Category
    weekly_store_category_df = aggregate_weekly_store_category_revenue(final_df)

    # Save processed dataset
    save_processed_data(
        weekly_store_category_df,
        "data/processed/weekly_store_category_revenue.csv"
    )

    print("=" * 60)
    print("PROCESSED WEEKLY STORE + CATEGORY REVENUE")
    print("=" * 60)
    print(weekly_store_category_df.head())
    print("\nShape:", weekly_store_category_df.shape)
