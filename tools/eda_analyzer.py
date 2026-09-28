import pandas as pd
import numpy as np
from tools.data_profiler import load_dataset


def analyze_eda(
    df: pd.DataFrame,
    id_columns:list = None
    ) -> dict:
    """
    Perform exploratory data analysis and return
    structured results that can later be consumed
    by the AI agent.
    """
    if id_columns is None:
        id_columns = []
    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "numerical_summary": {},
        "categorical_summary": {},
        "correlations": {},
        "strong_correlations": [],
        "outliers": {}
    }

    # --------------------------------------------------
    # 1. Identify column types
    # --------------------------------------------------

    numerical_columns = [
    column
    for column in df.select_dtypes(
        include=["number"]
    ).columns
    if column not in id_columns
]

    categorical_columns = [
    column
    for column in df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns
    if column not in id_columns
]

    # --------------------------------------------------
    # 2. Numerical Statistics
    # --------------------------------------------------

    for column in numerical_columns:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        report["numerical_summary"][column] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2)
        }

        # ----------------------------------------------
        # Outlier detection using IQR
        # ----------------------------------------------

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            ((series < lower_bound) |
             (series > upper_bound)).sum()
        )

        report["outliers"][column] = {
            "count": outlier_count,
            "percentage": round(
                (outlier_count / len(series)) * 100,
                2
            )
        }

    # --------------------------------------------------
    # 3. Categorical Statistics
    # --------------------------------------------------

    for column in categorical_columns:

        value_counts = (
            df[column]
            .value_counts(dropna=False)
        )

        total = len(df)

        categories = {}

        for value, count in value_counts.items():

            if pd.isna(value):
                value_name = "Missing"
            else:
                value_name = str(value)

            categories[value_name] = {
                "count": int(count),
                "percentage": round(
                    (count / total) * 100,
                    2
                )
            }

        report["categorical_summary"][column] = categories

    # --------------------------------------------------
    # 4. Correlation Matrix
    # --------------------------------------------------

    if len(numerical_columns) >= 2:

        correlation_matrix = (
            df[numerical_columns]
            .corr()
            .round(3)
        )

        for column in numerical_columns:

            report["correlations"][column] = {
                other_column: float(
                    correlation_matrix.loc[column, other_column]
                )
                for other_column in numerical_columns
            }

        # ----------------------------------------------
        # Find strong correlations
        # ----------------------------------------------

        for i in range(len(numerical_columns)):

            for j in range(i + 1, len(numerical_columns)):

                column_a = numerical_columns[i]
                column_b = numerical_columns[j]

                correlation = correlation_matrix.loc[
                    column_a,
                    column_b
                ]

                if not pd.isna(correlation):

                    report["strong_correlations"].append({
                        "feature_1": column_a,
                        "feature_2": column_b,
                        "correlation": float(correlation),
                        "absolute_correlation": float(
                            abs(correlation)
                        )
                    })

        report["strong_correlations"].sort(
            key=lambda x: x["absolute_correlation"],
            reverse=True
        )

    return report


def print_eda_report(report: dict):

    print("\n" + "=" * 60)
    print("              EDA ANALYSIS REPORT")
    print("=" * 60)

    # --------------------------------------------------
    # Dataset Shape
    # --------------------------------------------------

    shape = report["dataset_shape"]

    print("\nDataset:")
    print(f"  Rows: {shape['rows']}")
    print(f"  Columns: {shape['columns']}")

    # --------------------------------------------------
    # Numerical Summary
    # --------------------------------------------------

    print("\nNumerical Statistics:")

    for column, stats in report[
        "numerical_summary"
    ].items():

        print(f"\n  {column}:")

        print(f"    Mean:   {stats['mean']}")
        print(f"    Median: {stats['median']}")
        print(f"    Std:    {stats['std']}")
        print(f"    Min:    {stats['min']}")
        print(f"    Q1:     {stats['q1']}")
        print(f"    Q3:     {stats['q3']}")
        print(f"    Max:    {stats['max']}")

    # --------------------------------------------------
    # Categorical Summary
    # --------------------------------------------------

    print("\nCategorical Statistics:")

    for column, categories in report[
        "categorical_summary"
    ].items():

        print(f"\n  {column}:")

        for value, info in categories.items():

            print(
                f"    {value}: "
                f"{info['count']} "
                f"({info['percentage']}%)"
            )

    # --------------------------------------------------
    # Correlations
    # --------------------------------------------------

    print("\nStrongest Correlations:")

    strong_correlations = report[
        "strong_correlations"
    ]

    if strong_correlations:

        for item in strong_correlations[:10]:

            print(
                f"  {item['feature_1']} ↔ "
                f"{item['feature_2']}: "
                f"{item['correlation']:.3f}"
            )

    else:

        print("  No correlations available")

    # --------------------------------------------------
    # Outliers
    # --------------------------------------------------

    print("\nOutlier Analysis:")

    for column, info in report[
        "outliers"
    ].items():

        print(
            f"  {column}: "
            f"{info['count']} "
            f"({info['percentage']}%)"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":

    file_path = "data/cleaned_dataset.csv"

    df = load_dataset(file_path)

    report = analyze_eda(
        df,
        id_columns=["customer_id"]
        )

    print_eda_report(report)