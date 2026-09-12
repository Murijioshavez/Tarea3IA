"""Validation and conversion of user CSV data into Chronos-2 inputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MINIMUM_OBSERVATIONS = 3


class DatasetValidationError(ValueError):
    """An expected, user-correctable dataset or configuration problem."""


@dataclass(frozen=True)
class PreparedForecastData:
    context: pd.DataFrame
    historical: list[dict[str, object]]
    timestamp_column: str
    target_columns: list[str]
    horizon: int
    frequency: str


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a frame to JSON-ready records without leaking pandas timestamps."""
    result: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        result.append(
            {
                key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                for key, value in row.items()
            }
        )
    return result


def preview_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    if dataset.empty:
        raise DatasetValidationError("El archivo CSV no contiene filas.")
    if len(dataset.columns) == 0:
        raise DatasetValidationError("El archivo CSV no contiene columnas.")

    return {
        "row_count": len(dataset),
        "columns": [{"name": column, "dtype": str(dtype)} for column, dtype in dataset.dtypes.items()],
        "preview": json_records(dataset.head(10)),
    }


def _validate_configuration(
    dataset: pd.DataFrame,
    timestamp_column: object,
    target_columns: object,
    horizon: object,
) -> tuple[str, list[str], int]:
    if not isinstance(timestamp_column, str) or timestamp_column not in dataset.columns:
        raise DatasetValidationError("Selecciona una columna temporal existente.")
    if not isinstance(target_columns, list) or not target_columns or not all(isinstance(column, str) for column in target_columns):
        raise DatasetValidationError("Selecciona al menos una variable objetivo.")
    if len(set(target_columns)) != len(target_columns) or any(column not in dataset.columns for column in target_columns):
        raise DatasetValidationError("Las variables objetivo seleccionadas no son válidas.")
    if timestamp_column in target_columns:
        raise DatasetValidationError("La columna temporal no puede ser una variable objetivo.")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 1024:
        raise DatasetValidationError("El horizonte debe ser un entero entre 1 y 1024.")
    return timestamp_column, target_columns, horizon


def prepare_forecast_data(
    dataset: pd.DataFrame,
    *,
    timestamp_column: object,
    target_columns: object,
    horizon: object,
) -> PreparedForecastData:
    """Validate a single CSV time series and build Chronos-2's DataFrame input."""
    timestamp_column, target_columns, horizon = _validate_configuration(
        dataset, timestamp_column, target_columns, horizon
    )
    context = dataset[[timestamp_column, *target_columns]].copy()
    context[timestamp_column] = pd.to_datetime(context[timestamp_column], errors="coerce")
    if context[timestamp_column].isna().any():
        raise DatasetValidationError("La columna temporal contiene fechas inválidas o vacías.")
    if context[timestamp_column].duplicated().any():
        raise DatasetValidationError("La columna temporal contiene timestamps duplicados.")

    for column in target_columns:
        context[column] = pd.to_numeric(context[column], errors="coerce")
        if context[column].isna().any():
            raise DatasetValidationError(f"La variable '{column}' contiene valores faltantes o no numéricos.")

    context = context.sort_values(timestamp_column).reset_index(drop=True)
    if len(context) < MINIMUM_OBSERVATIONS:
        raise DatasetValidationError(
            f"Se requieren al menos {MINIMUM_OBSERVATIONS} observaciones para inferir la frecuencia temporal."
        )

    frequency = pd.infer_freq(context[timestamp_column])
    if frequency is None:
        raise DatasetValidationError("No se pudo inferir una frecuencia temporal regular en los timestamps.")

    historical = json_records(context)
    chronos_context = context.rename(columns={timestamp_column: "timestamp"})
    chronos_context.insert(0, "item_id", "uploaded-series")

    return PreparedForecastData(
        context=chronos_context,
        historical=historical,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        horizon=horizon,
        frequency=frequency,
    )
