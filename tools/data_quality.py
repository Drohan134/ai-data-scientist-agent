# Ye automatically detect karega:
# Missing values
# Missing %
# Duplicate rows
# ID columns
# Constant columns
# High-cardinality columns
# Binary columns
# Potential target columns
# Class imbalance


import pandas as pd


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """
    Analyze dataset quality and identify potential
    issues, ID columns, target candidates, and
    class distributions.
    """

    rows, columns = df.shape

    # -------------------------------------------------
    # 1. Missing Values
    # -------------------------------------------------

    missing = df.isnull().sum()

    missing_info = {}

    for column, count in missing.items():

        if count > 0:
            percentage = (count / rows) * 100

            missing_info[column] = {
                "count": int(count),
                "percentage": round(percentage, 2)
            }

    # -------------------------------------------------
    # 2. Duplicate Rows
    # -------------------------------------------------

    duplicate_rows = int(df.duplicated().sum())

    # -------------------------------------------------
    # 3. ID Column Detection
    # -------------------------------------------------

    id_columns = []

    for column in df.columns:

        unique_count = df[column].nunique(dropna=True)

        unique_ratio = unique_count / rows

        column_name = column.lower()

        # Common ID naming patterns
        id_keywords = [
            "id",
            "identifier",
            "customer_id",
            "user_id",
            "student_id",
            "employee_id"
        ]

        name_based_id = any(
            keyword == column_name
            or column_name.endswith("_" + keyword)
            for keyword in id_keywords
        )

        # High uniqueness can also indicate an ID
        high_uniqueness = unique_ratio >= 0.95
        
        is_string_column = str(df[column].dtype) in [
            "object",
            "string",
            "category"
        ]
        if name_based_id or (high_uniqueness and is_string_column):
            id_columns.append(column)

    # -------------------------------------------------
    # 4. Constant Columns
    # -------------------------------------------------

    constant_columns = []

    for column in df.columns:

        if df[column].nunique(dropna=True) <= 1:
            constant_columns.append(column)

    # -------------------------------------------------
    # 5. High Cardinality Columns
    # -------------------------------------------------

    high_cardinality_columns = []

    for column in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:

        unique_count = df[column].nunique(dropna=True)

        unique_ratio = unique_count / rows

        if unique_ratio >= 0.5 and column not in id_columns:
            high_cardinality_columns.append(column)

    # -------------------------------------------------
    # 6. Binary Columns
    # -------------------------------------------------

    binary_columns = []

    for column in df.columns:

        unique_count = df[column].nunique(dropna=True)

        if unique_count == 2:
            binary_columns.append(column)

    # -------------------------------------------------
    # 7. Potential Target Columns
    # -------------------------------------------------

    potential_targets = []

    for column in df.columns:

        # Ignore detected ID columns
        if column in id_columns:
            continue

        unique_count = df[column].nunique(dropna=True)
        dtype = str(df[column].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[column])

        # Binary target (any dtype)
        if unique_count == 2:
            potential_targets.append(column)

        # Small categorical target (string/category)
        elif (
            dtype in ["object", "string", "category"]
            and 2 <= unique_count <= 10
        ):
            potential_targets.append(column)

        # Numeric regression target: continuous column with many unique values
        # Use position heuristic: last numeric column with high unique count
        elif is_numeric and unique_count > 20:
            # Only add if it looks like a response variable (not a raw ID)
            unique_ratio = unique_count / max(rows, 1)
            if unique_ratio <= 0.98:   # not a near-perfect unique key
                potential_targets.append(column)

    # Prioritise: last column is most commonly the target in tabular datasets
    cols_list = df.columns.tolist()
    last_col = cols_list[-1] if cols_list else None
    if (
        last_col
        and last_col not in id_columns
        and last_col not in potential_targets
    ):
        potential_targets.append(last_col)

    # -------------------------------------------------
    # 8. Class Distribution
    # -------------------------------------------------

    class_distributions = {}

    for column in binary_columns:

        distribution = (
            df[column]
            .value_counts(normalize=True, dropna=False)
            .round(4)
            .to_dict()
        )

        class_distributions[column] = distribution

    # -------------------------------------------------
    # 9. Overall Quality Score
    # -------------------------------------------------

    quality_score = 100

    # Missing values penalty
    total_missing = int(df.isnull().sum().sum())

    if total_missing > 0:
        missing_percentage = (
            total_missing / (rows * columns)
        ) * 100

        quality_score -= min(
            missing_percentage * 2,
            30
        )

    # Duplicate penalty
    if duplicate_rows > 0:
        duplicate_percentage = (
            duplicate_rows / rows
        ) * 100

        quality_score -= min(
            duplicate_percentage * 2,
            20
        )

    # Constant column penalty
    quality_score -= len(constant_columns) * 5

    quality_score = max(
        0,
        round(quality_score, 2)
    )

    # -------------------------------------------------
    # Final Result
    # -------------------------------------------------

    quality_report = {

        "dataset_shape": {
            "rows": rows,
            "columns": columns
        },

        "missing_values": missing_info,

        "duplicate_rows": duplicate_rows,

        "id_columns": id_columns,

        "constant_columns": constant_columns,

        "high_cardinality_columns": high_cardinality_columns,

        "binary_columns": binary_columns,

        "potential_target_columns": potential_targets,

        "class_distributions": class_distributions,

        "quality_score": quality_score
    }

    return quality_report


def print_quality_report(report: dict):
    """
    Print the quality report in a human-readable format.
    """

    print("\n" + "=" * 60)
    print("             DATA QUALITY REPORT")
    print("=" * 60)

    # Missing values
    print("\nMissing Values:")

    if report["missing_values"]:

        for column, info in report["missing_values"].items():

            print(
                f"  ⚠ {column}: "
                f"{info['count']} "
                f"({info['percentage']}%)"
            )

    else:
        print("  ✓ No missing values")

    # Duplicate rows
    print(
        f"\nDuplicate Rows: "
        f"{report['duplicate_rows']}"
    )

    # ID columns
    print("\nPotential ID Columns:")

    if report["id_columns"]:

        for column in report["id_columns"]:
            print(f"  ℹ {column}")

    else:
        print("  None detected")

    # Constant columns
    print("\nConstant Columns:")

    if report["constant_columns"]:

        for column in report["constant_columns"]:
            print(f"  ⚠ {column}")

    else:
        print("  ✓ None")

    # High cardinality
    print("\nHigh Cardinality Columns:")

    if report["high_cardinality_columns"]:

        for column in report["high_cardinality_columns"]:
            print(f"  ⚠ {column}")

    else:
        print("  ✓ None")

    # Binary columns
    print("\nBinary Columns:")

    for column in report["binary_columns"]:
        print(f"  - {column}")

    # Potential targets
    print("\nPotential Target Columns:")

    if report["potential_target_columns"]:

        for column in report["potential_target_columns"]:
            print(f"  🎯 {column}")

    else:
        print("  None detected")

    # Class distributions
    print("\nClass Distributions:")

    for column, distribution in report[
        "class_distributions"
    ].items():

        print(f"\n  {column}:")

        for value, percentage in distribution.items():

            print(
                f"    {value}: "
                f"{percentage * 100:.2f}%"
            )

    # Quality score
    print(
        f"\nData Quality Score: "
        f"{report['quality_score']}/100"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":

    # Import profiler
    from tools.data_profiler import load_dataset

    file_path = "data/test_dataset.csv"

    df = load_dataset(file_path)

    report = analyze_data_quality(df)

    print_quality_report(report)