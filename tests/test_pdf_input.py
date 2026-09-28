import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.pdf_extractor import extract_pdf_table


pdf_path = "data/table.pdf"

df = extract_pdf_table(pdf_path)

print("\nPDF EXTRACTION TEST")
print("=" * 50)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column Names:")
print(list(df.columns))

print("\nFirst 5 Rows:")
print(df.head())