import re
from pathlib import Path

import pandas as pd
import pdfplumber


def _clean_text(value):
    if value is None:
        return ""

    text = str(value).replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def _non_empty(value):
    return _clean_text(value) != ""


def _make_unique_columns(columns):
    result = []
    counts = {}

    for col in columns:
        col = _clean_text(col) or "Unnamed"
        counts[col] = counts.get(col, 0) + 1

        if counts[col] == 1:
            result.append(col)
        else:
            result.append(f"{col}_{counts[col]}")

    return result

def _infer_column_types(df):
    """
    Convert PDF-extracted numeric values to numeric dtype
    when the whole column contains simple numeric values.

    Complex text values such as:
    '98.3% n=2 (97.7%, n=3)'
    remain strings.
    """

    for column in df.columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Convert only when every non-empty value
        # is a simple numeric value.
        non_empty = df[column].notna()

        if (
            non_empty.any()
            and converted[non_empty].notna().all()
        ):
            df[column] = converted

    return df

def _find_data_start(rows):
    """
    Find the first row that looks like a real data row.
    """

    counts = [
        sum(_non_empty(value) for value in row)
        for row in rows
    ]

    if not counts:
        return None, None

    width = max(counts)

    if width < 2:
        return None, None

    # Allow one missing value in a data row.
    threshold = max(2, width - 1)

    for i, count in enumerate(counts):
        if count >= threshold:
            return i, width

    return None, None


def _extract_header(page, table, logical_cells, data_top):
    """
    Reconstruct headers from PDF words.

    Handles:
    - multi-line headers
    - merged headers
    - grouped headers
    """

    ranges = [
        (cell[0], cell[2])
        for cell in logical_cells
    ]

    headers = [[] for _ in ranges]

    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=2
    )

    for word in words:

        text = _clean_text(word.get("text"))

        if not text:
            continue

        cx = (word["x0"] + word["x1"]) / 2
        cy = (word["top"] + word["bottom"]) / 2

        # Only look above the first data row.
        if not (table.bbox[1] <= cy < data_top):
            continue

        # Find the PDF table cell containing this word.
        owner = None

        for cell in table.cells:

            if cell is None:
                continue

            x0, top, x1, bottom = cell

            if (
                x0 <= cx <= x1
                and top <= cy <= bottom
            ):
                owner = cell
                break

        if owner is None:
            continue

        # Ignore group headers that span multiple
        # logical columns, e.g. "Results".
        overlap_count = 0

        for x0, x1 in ranges:

            overlap = max(
                0,
                min(owner[2], x1)
                - max(owner[0], x0)
            )

            if overlap > 1:
                overlap_count += 1

        if overlap_count > 1:
            continue

        # Assign the word to its logical column.
        column_index = None

        for i, (x0, x1) in enumerate(ranges):

            if x0 <= cx <= x1:
                column_index = i
                break

        if column_index is not None:

            headers[column_index].append(
                (
                    word["top"],
                    word["x0"],
                    text
                )
            )

    final_headers = []

    for parts in headers:

        # Preserve reading order.
        parts.sort(
            key=lambda item: (
                item[0],
                item[1]
            )
        )

        final_headers.append(
            " ".join(
                item[2]
                for item in parts
            )
        )

    return _make_unique_columns(final_headers)


def _extract_table(page, table):

    rows = table.extract()

    if not rows:
        return None

    data_start, width = _find_data_start(rows)

    if data_start is None:
        return None

    # Physical columns containing the actual data.
    first_data_row = rows[data_start]

    logical_indexes = [
        i
        for i, value in enumerate(first_data_row)
        if _non_empty(value)
    ]

    if len(logical_indexes) < 2:
        return None

    # Get actual cell coordinates for those columns.
    data_row_cells = [
        cell
        for cell in table.rows[data_start].cells
        if cell is not None
    ]

    if len(data_row_cells) != len(logical_indexes):
        return None

    # Reconstruct header.
    headers = _extract_header(
        page,
        table,
        data_row_cells,
        data_row_cells[0][1]
    )

    data = []

    for row in rows[data_start:]:

        values = []

        for index in logical_indexes:

            value = (
                row[index]
                if index < len(row)
                else None
            )

            values.append(
                _clean_text(value) or None
            )

        # Skip completely empty rows.
        if any(
            value is not None
            for value in values
        ):
            data.append(values)

    if not data:
        return None

    df = pd.DataFrame(
        data,
        columns=headers
    )

    # Remove completely empty rows/columns.
    df = df.dropna(
        axis=0,
        how="all"
    )

    df = df.dropna(
        axis=1,
        how="all"
    )

    return df.reset_index(drop=True)


def extract_pdf_table(pdf_path):
    """
    Extract the main tabular data from a PDF.

    Handles:
    - multi-row headers
    - merged/grouped headers
    - multiline headers
    - multiline data cells
    - empty spacer rows
    - multiple PDF pages

    Returns:
        pandas.DataFrame
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    extracted_tables = []

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            tables = page.find_tables()

            for table in tables:

                df = _extract_table(
                    page,
                    table
                )

                if (
                    df is not None
                    and not df.empty
                ):
                    extracted_tables.append(df)

    if not extracted_tables:
        raise ValueError(
        "No usable tabular data found in the PDF."
    )

    if len(extracted_tables) == 1:
        return _infer_column_types(
        extracted_tables[0]
    )

    # Combine multi-page tables when
    # their schemas are identical.
    first_columns = list(
        extracted_tables[0].columns
    )

    if all(
        list(df.columns) == first_columns
        for df in extracted_tables
    ):
        return pd.concat(
            extracted_tables,
            ignore_index=True
        )
        return _infer_column_types(result)

    # If unrelated tables exist,
    # return the largest usable table.
    return max(
        extracted_tables,
        key=lambda df:
        df.shape[0] * df.shape[1]
    )
    return _infer_column_types(result)
    
