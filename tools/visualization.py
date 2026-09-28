import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tools.data_profiler import load_dataset


OUTPUT_DIR = "reports/visualizations"

def safe_filename(name):
    """Convert a column name into a Windows-safe filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', str(name))

def create_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# Visualization safety limits.
# Visualization should never try to render hundreds of thousands of points.
SMALL_DATASET_ROWS = 10_000
MAX_PLOT_ROWS = 5_000
MAX_CATEGORICAL_CATEGORIES = 20
MAX_CORRELATION_COLUMNS = 30
MAX_DISTRIBUTION_PLOTS = 12


def prepare_visualization_data(df: pd.DataFrame):
    """Return a bounded visualization dataframe without modifying the source."""
    row_count = len(df)

    if row_count <= SMALL_DATASET_ROWS:
        return df, {
            "rows": row_count,
            "strategy": "full_dataset",
            "sample_rows": row_count,
        }

    sample_size = min(MAX_PLOT_ROWS, row_count)
    sample_df = df.sample(n=sample_size, random_state=42)

    return sample_df, {
        "rows": row_count,
        "strategy": "representative_sample",
        "sample_rows": sample_size,
    }


def _correlation_columns(numerical_columns: list):
    return numerical_columns[:MAX_CORRELATION_COLUMNS]


def plot_numerical_distributions(
    df: pd.DataFrame,
    columns: list
):
    """
    Create histograms for numerical columns.

    KDE is intentionally disabled for large datasets because it can be
    significantly slower than a plain histogram.
    """

    paths = []
    use_kde = len(df) <= SMALL_DATASET_ROWS

    for column in columns:

        plt.figure(figsize=(8, 5))

        values = df[column].dropna()

        sns.histplot(
            values,
            kde=use_kde,
            bins=30
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        path = os.path.join(
            OUTPUT_DIR,
            f"{safe_filename(column)}_distribution.png"
        )

        plt.savefig(path, dpi=100)
        plt.close()

        paths.append(path)

    return paths


def plot_categorical_distributions(
    df: pd.DataFrame,
    columns: list
):
    """
    Create bar charts for categorical columns.

    Counts use the bounded visualization sample for large datasets.
    """

    paths = []

    for column in columns:

        plt.figure(figsize=(8, 5))

        value_counts = (
            df[column]
            .value_counts(dropna=False)
        )

        if len(value_counts) > MAX_CATEGORICAL_CATEGORIES:
            value_counts = value_counts.head(MAX_CATEGORICAL_CATEGORIES)

        sns.barplot(
            x=value_counts.index.astype(str),
            y=value_counts.values
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")

        plt.xticks(rotation=30)

        plt.tight_layout()

        path = os.path.join(
            OUTPUT_DIR,
            f"{safe_filename(column)}_distribution.png"
        )

        plt.savefig(path, dpi=100)
        plt.close()

        paths.append(path)

    return paths


def plot_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str
):
    """
    Create a scatter plot between two numerical columns.
    """

    # Never render hundreds of thousands of points.
    plot_df, _ = prepare_visualization_data(df)

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=plot_df,
        x=x_column,
        y=y_column
    )

    plt.title(
        f"{x_column} vs {y_column}"
    )

    plt.tight_layout()

    filename = (
        f"{x_column}_vs_{y_column}.png"
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(path)
    plt.close()

    return path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list
):
    """
    Create correlation heatmap.
    """

    columns = columns[:MAX_CORRELATION_COLUMNS]
    correlation = df[columns].corr()

    plt.figure(
        figsize=(10, 8)
    )

    # Annotation becomes cluttered/slow for large correlation matrices.
    annotate = len(columns) <= 15

    sns.heatmap(
        correlation,
        annot=annotate,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    )

    plt.savefig(path)
    plt.close()

    return path


def generate_visualizations(df: pd.DataFrame):

    create_output_directory()

    numerical_columns = (
        df.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=["object", "string", "category", "bool"]
        )
        .columns
        .tolist()
    )

    # Remove obvious ID column.
    id_columns = ["customer_id"]

    numerical_columns = [
        column for column in numerical_columns
        if column not in id_columns
    ]

    categorical_columns = [
        column for column in categorical_columns
        if column not in id_columns
    ]

    plot_df, strategy = prepare_visualization_data(df)

    print(
        f"Visualization strategy: {strategy['strategy']} "
        f"({strategy['sample_rows']} of {strategy['rows']} rows)"
    )

    if strategy["rows"] > SMALL_DATASET_ROWS:
        print(
            f"Large dataset detected: row-level plots use at most "
            f"{MAX_PLOT_ROWS} sampled rows; categorical plots use "
            f"full-data counts with top {MAX_CATEGORICAL_CATEGORIES} categories."
        )

    generated_files = []

    # Expensive row-level plots use a bounded sample for large datasets.
    generated_files.extend(
        plot_numerical_distributions(
            plot_df,
            numerical_columns
        )
    )

    # Keep chart generation bounded for wide data and large uploads.
    numerical_columns = numerical_columns[:MAX_DISTRIBUTION_PLOTS]
    categorical_columns = categorical_columns[:MAX_DISTRIBUTION_PLOTS]

    generated_files.extend(
        plot_categorical_distributions(
            plot_df,
            categorical_columns
        )
    )

    # Correlation uses the bounded sample and max 30 numerical columns.
    if len(numerical_columns) >= 2:
        heatmap_path = plot_correlation_heatmap(
            plot_df,
            _correlation_columns(numerical_columns)
        )
        generated_files.append(heatmap_path)

    return generated_files


if __name__ == "__main__":

    file_path = "data/cleaned_dataset.csv"

    df = load_dataset(file_path)

    files = generate_visualizations(df)

    print("\n" + "=" * 60)
    print("          VISUALIZATION REPORT")
    print("=" * 60)

    print(
        f"\nGenerated {len(files)} visualizations:\n"
    )

    for file in files:
        print(f"  ✓ {file}")

    print("\n" + "=" * 60)