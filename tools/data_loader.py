import pandas as pd
from tools.pdf_extractor import extract_pdf_table


def load_dataset(file_path):
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path)

    if file_path.lower().endswith(".pdf"):
        return extract_pdf_table(file_path)

    raise ValueError("Unsupported file type. Use CSV or PDF.")