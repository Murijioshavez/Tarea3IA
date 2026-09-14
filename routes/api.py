"""JSON endpoints for dataset upload, preview, and forecasting."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from services.data_processing import DatasetValidationError, prepare_forecast_data, preview_dataset
from services.forecasting import ForecastingError, forecast


api = Blueprint("api", __name__, url_prefix="/api")
ALLOWED_SUFFIXES = {".csv"}
UPLOAD_MAX_AGE_SECONDS = 24 * 60 * 60


def prune_uploads(folder: Path) -> None:
    """Drop uploads older than a day; nothing else ever deletes a successful upload."""
    cutoff = time.time() - UPLOAD_MAX_AGE_SECONDS
    for path in folder.glob("*.csv"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            current_app.logger.warning("No se pudo eliminar el archivo temporal %s", path.name)


def error_response(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def dataset_path(dataset_id: str) -> Path:
    """Return a safe upload path only for canonical UUID identifiers."""
    try:
        canonical_id = str(UUID(dataset_id))
    except (TypeError, ValueError):
        raise DatasetValidationError("El identificador del dataset no es válido.") from None

    return Path(current_app.config["UPLOAD_FOLDER"]) / f"{canonical_id}.csv"


def load_dataset(dataset_id: str) -> pd.DataFrame:
    path = dataset_path(dataset_id)
    if not path.is_file():
        raise DatasetValidationError("No se encontró el dataset solicitado.")

    # CSVs exported from Excel or downloaded from Kaggle are often CP1252/Latin-1 rather
    # than UTF-8. Latin-1 decodes any byte sequence, so it always terminates the loop.
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise DatasetValidationError("El archivo CSV está vacío.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetValidationError(
                "No se pudo interpretar el CSV. Revisa que use comas como separador y que "
                "todas las filas tengan el mismo número de columnas."
            ) from exc
        except OSError as exc:
            raise DatasetValidationError("No se pudo leer el archivo CSV.") from exc

    raise DatasetValidationError("No se pudo determinar la codificación del archivo CSV.")


@api.post("/upload")
def upload_dataset():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return error_response("Selecciona un archivo CSV para cargar.")

    filename = secure_filename(uploaded_file.filename)
    if not filename or Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        return error_response("Solo se permiten archivos con extensión .csv.")

    prune_uploads(Path(current_app.config["UPLOAD_FOLDER"]))

    dataset_id = str(uuid4())
    path = dataset_path(dataset_id)
    uploaded_file.save(path)

    try:
        dataset = load_dataset(dataset_id)
        preview = preview_dataset(dataset)
    except DatasetValidationError as exc:
        path.unlink(missing_ok=True)
        return error_response(str(exc))

    return jsonify({"dataset_id": dataset_id, **preview}), 201


@api.get("/datasets/<dataset_id>/preview")
def dataset_preview(dataset_id: str):
    try:
        return jsonify({"dataset_id": dataset_id, **preview_dataset(load_dataset(dataset_id))})
    except DatasetValidationError as exc:
        return error_response(str(exc), 404 if "encontró" in str(exc) else 400)


@api.post("/forecast")
def run_forecast():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error_response("Envía la configuración del forecast como JSON.")

    try:
        prepared = prepare_forecast_data(
            load_dataset(payload.get("dataset_id")),
            timestamp_column=payload.get("timestamp_column"),
            target_columns=payload.get("target_columns"),
            covariate_columns=payload.get("covariate_columns"),
            horizon=payload.get("horizon"),
        )
        result = forecast(prepared)
    except DatasetValidationError as exc:
        current_app.logger.info("Forecast rechazado: %s", exc)
        return error_response(str(exc))
    except ForecastingError as exc:
        current_app.logger.exception("Chronos-2 inference failed")
        return error_response(str(exc), 500)

    return jsonify(result)
