import pandas as pd


def inspect_excel(path):
    xls = pd.ExcelFile(path)

    print("=" * 50)
    print("AVAILABLE SHEETS")
    print("=" * 50)
    print(xls.sheet_names)

    print("\n")

    for sheet in xls.sheet_names:
        print("=" * 50)
        print(f"SHEET: {sheet}")
        print("=" * 50)

        df = pd.read_excel(path, sheet_name=sheet)

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Missing Values:\n", df.isnull().sum())
        print("First 3 Rows:\n", df.head(3))
        print("\n\n")


if __name__ == "__main__":
    inspect_excel("data/raw/retail_sales_dataset.xlsx")
