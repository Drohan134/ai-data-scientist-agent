import pandas as pd
from tools.data_profiler import load_dataset


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean dataset and return:
    1. Cleaned DataFrame
    2. Cleaning report
    """

    cleaned_df = df.copy()

    report = {
        "rows_before": len(cleaned_df),
        "rows_after": None,
        "missing_values_handled": {},
        "duplicates_removed": 0,
        "constant_columns_removed": [],
        "actions": []
    }

    # --------------------------------------------------
    # 1. Handle Missing Values
    # --------------------------------------------------

    missing_columns = cleaned_df.columns[
        cleaned_df.isnull().any()
    ].tolist()

    for column in missing_columns:

        missing_count = int(cleaned_df[column].isnull().sum())

        if pd.api.types.is_numeric_dtype(cleaned_df[column]):

            median_value = cleaned_df[column].median()

            cleaned_df[column] = cleaned_df[column].fillna(
                median_value
            )

            report["missing_values_handled"][column] = {
                "missing_count": missing_count,
                "method": "median_imputation",
                "value": float(median_value)
            }

            report["actions"].append(
                f"Imputed {missing_count} missing value(s) "
                f"in '{column}' using median ({median_value})"
            )

        else:

            mode_values = cleaned_df[column].mode()

            if not mode_values.empty:

                mode_value = mode_values.iloc[0]

                cleaned_df[column] = cleaned_df[column].fillna(
                    mode_value
                )

                report["missing_values_handled"][column] = {
                    "missing_count": missing_count,
                    "method": "mode_imputation",
                    "value": str(mode_value)
                }

                report["actions"].append(
                    f"Imputed {missing_count} missing value(s) "
                    f"in '{column}' using mode ('{mode_value}')"
                )

    # --------------------------------------------------
    # 2. Remove Duplicate Rows
    # --------------------------------------------------

    duplicate_count = int(cleaned_df.duplicated().sum())

    if duplicate_count > 0:

        cleaned_df = cleaned_df.drop_duplicates()

        report["duplicates_removed"] = duplicate_count

        report["actions"].append(
            f"Removed {duplicate_count} duplicate row(s)"
        )

    # --------------------------------------------------
    # 3. Detect Constant Columns
    # --------------------------------------------------

    constant_columns = []

    for column in cleaned_df.columns:

        if cleaned_df[column].nunique(dropna=True) <= 1:
            constant_columns.append(column)

    # We don't automatically remove ID columns or target columns.
    # Constant columns are removed because they provide no information.

    if constant_columns:

        cleaned_df = cleaned_df.drop(
            columns=constant_columns
        )

        report["constant_columns_removed"] = constant_columns

        for column in constant_columns:
            report["actions"].append(
                f"Removed constant column '{column}'"
            )

    # --------------------------------------------------
    # 4. Final Information
    # --------------------------------------------------

    report["rows_after"] = len(cleaned_df)

    report["columns_before"] = len(df.columns)

    report["columns_after"] = len(cleaned_df.columns)

    report["remaining_missing_values"] = int(
        cleaned_df.isnull().sum().sum()
    )

    return cleaned_df, report


def print_cleaning_report(report: dict):

    print("\n" + "=" * 60)
    print("              DATA CLEANING REPORT")
    print("=" * 60)

    print(
        f"\nRows Before: {report['rows_before']}"
    )

    print(
        f"Rows After:  {report['rows_after']}"
    )

    print(
        f"\nColumns Before: {report['columns_before']}"
    )

    print(
        f"Columns After:  {report['columns_after']}"
    )

    print("\nMissing Values Handled:")

    if report["missing_values_handled"]:

        for column, info in report[
            "missing_values_handled"
        ].items():

            print(
                f"  ✓ {column}: "
                f"{info['missing_count']} missing → "
                f"{info['method']}"
            )

    else:

        print("  ✓ No missing values")

    print(
        f"\nDuplicate Rows Removed: "
        f"{report['duplicates_removed']}"
    )

    print("\nConstant Columns Removed:")

    if report["constant_columns_removed"]:

        for column in report[
            "constant_columns_removed"
        ]:

            print(f"  ✓ {column}")

    else:

        print("  ✓ None")

    print(
        f"\nRemaining Missing Values: "
        f"{report['remaining_missing_values']}"
    )

    print("\nCleaning Actions:")

    if report["actions"]:

        for action in report["actions"]:
            print(f"  ✓ {action}")

    else:

        print("  ✓ No cleaning required")

    print("\n" + "=" * 60)


if __name__ == "__main__":

    file_path = "data/test_dataset.csv"

    df = load_dataset(file_path)

    cleaned_df, report = clean_data(df)

    print_cleaning_report(report)

    # Save cleaned dataset
    output_path = "data/cleaned_dataset.csv"

    cleaned_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nCleaned dataset saved to: "
        f"{output_path}"
    )