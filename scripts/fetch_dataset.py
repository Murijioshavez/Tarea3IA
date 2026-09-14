"""Download the Kaggle dataset and write a CSV the app can ingest directly.

The raw file has no timestamp column, so it cannot be uploaded as-is: Chronos-2 needs a
regular time index. This script attaches a synthetic daily index over the rows in file
order and keeps only the financial columns, which is exactly the layout the forecast
endpoint expects.

    python scripts/fetch_dataset.py
    python scripts/fetch_dataset.py --output data/cars.csv --keep-identifiers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "mohdshahnawazaadil/sales-prediction-dataset"
SOURCE_FILE = "car_purchasing.csv"
TIMESTAMP_COLUMN = "fecha"
TARGET = "car purchase amount"

# Financial and demographic columns only. Name, e-mail and country are identifiers: the
# first two are unique per row and country has 211 distinct values across 500 rows, so
# none of them carry usable signal.
FINANCIAL_COLUMNS = ["age", "annual Salary", "credit card debt", "net worth", "gender"]
IDENTIFIER_COLUMNS = ["customer name", "customer e-mail", "country"]


def fetch() -> pd.DataFrame:
    """Load the dataset straight into pandas; the file is Latin-1, not UTF-8."""
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        DATASET,
        SOURCE_FILE,
        pandas_kwargs={"encoding": "latin-1"},
    )


def build(dataset: pd.DataFrame, *, keep_identifiers: bool) -> pd.DataFrame:
    columns = [TARGET, *FINANCIAL_COLUMNS]
    if keep_identifiers:
        columns += [column for column in IDENTIFIER_COLUMNS if column in dataset.columns]

    frame = dataset[columns].copy()
    frame.insert(0, TIMESTAMP_COLUMN, pd.date_range("2020-01-01", periods=len(frame), freq="D"))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "car_purchasing.csv")
    parser.add_argument(
        "--keep-identifiers",
        action="store_true",
        help="Conserva nombre, correo y país (no aportan señal; útil solo para inspección).",
    )
    arguments = parser.parse_args()

    dataset = fetch()
    print(f"Descargado '{SOURCE_FILE}' de {DATASET}: {len(dataset)} filas × {len(dataset.columns)} columnas")

    frame = build(dataset, keep_identifiers=arguments.keep_identifiers)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output, index=False)

    print(f"Guardado en {arguments.output}: {len(frame)} filas × {len(frame.columns)} columnas")
    print(f"\nEn la aplicación selecciona:")
    print(f"  Columna temporal      : {TIMESTAMP_COLUMN}")
    print(f"  Variable a pronosticar: {TARGET}")
    print(f"  Covariables           : {', '.join(FINANCIAL_COLUMNS)}")


if __name__ == "__main__":
    main()
