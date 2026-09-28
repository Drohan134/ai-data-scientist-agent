# Iska kaam hoga kisi bhi CSV/XLSX ko read karke structured dataset profile return karna.

import pandas as pd
from pathlib import Path

from tools.pdf_extractor import extract_pdf_table

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load CSV, Excel, or PDF dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(path)

    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(path)

    elif extension == ".pdf":
        df = extract_pdf_table(path)

    else:
        raise ValueError(
            "Unsupported file format. "
            "Please use CSV, Excel, or PDF file."
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a structured profile of the dataset.
    """

    # Basic information
    rows, columns = df.shape

    # Column type detection
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
    include=["object", "string", "category", "bool"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    # Missing values
    missing_values = df.isnull().sum()

    missing_values = {
        column: int(count)
        for column, count in missing_values.items()
        if count > 0
    }

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Unique values
    unique_values = {
        column: int(df[column].nunique(dropna=True))
        for column in df.columns
    }

    # Dataset memory
    memory_usage_mb = round(
        df.memory_usage(deep=True).sum() / (1024 ** 2),
        2
    )

    # Basic statistics
    numerical_statistics = {}

    if numerical_columns:
        stats = df[numerical_columns].describe()

        numerical_statistics = stats.to_dict()

    profile = {
        "shape": {
            "rows": rows,
            "columns": columns
        },

        "columns": df.columns.tolist(),

        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "numerical_columns": numerical_columns,

        "categorical_columns": categorical_columns,

        "datetime_columns": datetime_columns,

        "missing_values": missing_values,

        "duplicate_rows": duplicate_rows,

        "unique_values": unique_values,

        "memory_usage_mb": memory_usage_mb,

        "numerical_statistics": numerical_statistics
    }

    return profile


def print_profile(profile: dict):
    """
    Display dataset profile in a readable format.
    """

    print("\n" + "=" * 60)
    print("             AI DATA SCIENTIST")
    print("             DATASET PROFILE")
    print("=" * 60)

    shape = profile["shape"]

    print(f"\nRows              : {shape['rows']}")
    print(f"Columns           : {shape['columns']}")

    print("\nNumerical Columns:")
    for column in profile["numerical_columns"]:
        print(f"  - {column}")

    print("\nCategorical Columns:")
    for column in profile["categorical_columns"]:
        print(f"  - {column}")

    print("\nDatetime Columns:")
    for column in profile["datetime_columns"]:
        print(f"  - {column}")

    print("\nMissing Values:")

    if profile["missing_values"]:
        for column, count in profile["missing_values"].items():
            print(f"  - {column}: {count}")
    else:
        print("  None")

    print(f"\nDuplicate Rows   : {profile['duplicate_rows']}")

    print("\nUnique Values:")

    for column, count in profile["unique_values"].items():
        print(f"  - {column}: {count}")

    print(f"\nMemory Usage     : {profile['memory_usage_mb']} MB")

    print("\n" + "=" * 60)


if __name__ == "__main__":

    file_path = "data/table.pdf"

    df = load_dataset(file_path)

    profile = profile_dataset(df)

    print_profile(profile)