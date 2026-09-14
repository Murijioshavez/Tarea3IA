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
    covariate_columns: list[str]
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


def infer_frequency(timestamps: pd.Series) -> str:
    """Resolve the series frequency, falling back to the most common gap.

    ``pd.infer_freq`` only succeeds on perfectly regular timestamps, so a single missing
    day or a calendar gap makes it return None. Real CSVs have gaps, so rather than
    rejecting the upload the dominant spacing is used instead.
    """
    inferred = pd.infer_freq(timestamps)
    if inferred is not None:
        return inferred

    gaps = timestamps.diff().dropna()
    if gaps.empty:
        raise DatasetValidationError("No se pudo inferir una frecuencia temporal en los timestamps.")

    dominant = gaps.mode().iloc[0]
    if dominant <= pd.Timedelta(0):
        raise DatasetValidationError("Los timestamps no están ordenados de forma creciente.")
    if (gaps == dominant).mean() < 0.5:
        raise DatasetValidationError(
            "Los timestamps son demasiado irregulares para inferir una frecuencia. "
            "Agrega o reindexa los datos a intervalos constantes."
        )
    return pd.tseries.frequencies.to_offset(dominant).freqstr


def timestamp_candidates(dataset: pd.DataFrame) -> list[str]:
    """List the columns usable as the time axis: real dates, one row each.

    Reported at preview time so a CSV without any time dimension is flagged immediately
    instead of failing later, once the user has already picked targets and a horizon.
    """
    candidates: list[str] = []
    for column in dataset.columns:
        values = dataset[column]
        if pd.api.types.is_numeric_dtype(values):
            continue
        try:
            parsed = pd.to_datetime(values, errors="coerce")
        except (ValueError, TypeError):
            continue
        if parsed.isna().any() or parsed.duplicated().any():
            continue
        candidates.append(column)
    return candidates


def preview_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    if dataset.empty:
        raise DatasetValidationError("El archivo CSV no contiene filas.")
    if len(dataset.columns) == 0:
        raise DatasetValidationError("El archivo CSV no contiene columnas.")

    return {
        "row_count": len(dataset),
        "columns": [{"name": column, "dtype": str(dtype)} for column, dtype in dataset.dtypes.items()],
        "timestamp_candidates": timestamp_candidates(dataset),
        "preview": json_records(dataset.head(10)),
    }


def _validate_configuration(
    dataset: pd.DataFrame,
    timestamp_column: object,
    target_columns: object,
    covariate_columns: object,
    horizon: object,
) -> tuple[str, list[str], list[str], int]:
    if not isinstance(timestamp_column, str) or timestamp_column not in dataset.columns:
        raise DatasetValidationError("Selecciona una columna temporal existente.")
    if not isinstance(target_columns, list) or not target_columns or not all(isinstance(column, str) for column in target_columns):
        raise DatasetValidationError("Selecciona al menos una variable objetivo.")
    if len(set(target_columns)) != len(target_columns) or any(column not in dataset.columns for column in target_columns):
        raise DatasetValidationError("Las variables objetivo seleccionadas no son válidas.")
    if timestamp_column in target_columns:
        raise DatasetValidationError("La columna temporal no puede ser una variable objetivo.")

    if covariate_columns is None:
        covariate_columns = []
    if not isinstance(covariate_columns, list) or not all(isinstance(column, str) for column in covariate_columns):
        raise DatasetValidationError("Las covariables deben enviarse como una lista de nombres de columna.")
    if len(set(covariate_columns)) != len(covariate_columns) or any(column not in dataset.columns for column in covariate_columns):
        raise DatasetValidationError("Las covariables seleccionadas no son válidas.")
    overlapping = set(covariate_columns) & ({timestamp_column} | set(target_columns))
    if overlapping:
        raise DatasetValidationError(
            f"Una covariable no puede ser también columna temporal u objetivo: {', '.join(sorted(overlapping))}."
        )

    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 1024:
        raise DatasetValidationError("El horizonte debe ser un entero entre 1 y 1024.")
    return timestamp_column, target_columns, covariate_columns, horizon


def prepare_forecast_data(
    dataset: pd.DataFrame,
    *,
    timestamp_column: object,
    target_columns: object,
    covariate_columns: object = None,
    horizon: object,
) -> PreparedForecastData:
    """Validate a single CSV time series and build Chronos-2's DataFrame input.

    Covariates are kept alongside the targets so Chronos-2 can condition on them;
    dropping them is what limits the model to whatever signal the target carries alone.
    """
    timestamp_column, target_columns, covariate_columns, horizon = _validate_configuration(
        dataset, timestamp_column, target_columns, covariate_columns, horizon
    )
    context = dataset[[timestamp_column, *target_columns, *covariate_columns]].copy()

    # A purely numeric column silently converts to epoch nanoseconds, producing timestamps
    # in 1970 instead of an error. Reject it up front so the failure names the real problem.
    if pd.api.types.is_numeric_dtype(context[timestamp_column]):
        raise DatasetValidationError(
            f"La columna '{timestamp_column}' es numérica, no una fecha. "
            "Selecciona una columna con fechas reales (por ejemplo 2020-01-31); si el CSV no "
            "tiene ninguna, no es una serie temporal y no puede pronosticarse."
        )

    context[timestamp_column] = pd.to_datetime(context[timestamp_column], errors="coerce")
    if context[timestamp_column].isna().any():
        raise DatasetValidationError(
            f"La columna '{timestamp_column}' contiene fechas inválidas o vacías."
        )
    duplicates = context[timestamp_column].duplicated().sum()
    if duplicates:
        raise DatasetValidationError(
            f"La columna '{timestamp_column}' tiene {duplicates} fechas repetidas y cada "
            "observación debe tener una fecha única. Si el CSV no tiene una columna de "
            "fechas, no es una serie temporal y no puede pronosticarse."
        )

    for column in [*target_columns, *covariate_columns]:
        context[column] = pd.to_numeric(context[column], errors="coerce")
        if context[column].isna().any():
            raise DatasetValidationError(f"La variable '{column}' contiene valores faltantes o no numéricos.")

    context = context.sort_values(timestamp_column).reset_index(drop=True)
    if len(context) < MINIMUM_OBSERVATIONS:
        raise DatasetValidationError(
            f"Se requieren al menos {MINIMUM_OBSERVATIONS} observaciones para inferir la frecuencia temporal."
        )
    if len(context) <= horizon + MINIMUM_OBSERVATIONS:
        raise DatasetValidationError(
            "No hay suficientes observaciones para separar entrenamiento y validación. "
            f"Usa un horizonte menor que {len(context) - MINIMUM_OBSERVATIONS}."
        )

    frequency = infer_frequency(context[timestamp_column])

    historical = json_records(context)
    chronos_context = context.rename(columns={timestamp_column: "timestamp"})
    chronos_context.insert(0, "item_id", "uploaded-series")

    return PreparedForecastData(
        context=chronos_context,
        historical=historical,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        covariate_columns=covariate_columns,
        horizon=horizon,
        frequency=frequency,
    )
