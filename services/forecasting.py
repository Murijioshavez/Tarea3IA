"""Thin adapter around the installed Chronos-2 forecasting API."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))

from chronos import Chronos2Pipeline

from services.data_processing import PreparedForecastData


MODEL_ID = "amazon/chronos-2"
QUANTILES = [0.1, 0.5, 0.9]
_pipeline: Chronos2Pipeline | None = None
_pipeline_lock = Lock()


class ForecastingError(RuntimeError):
    """An inference failure that should not expose internal details to clients."""


def get_pipeline() -> Chronos2Pipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")
    return _pipeline


def forecast(prepared: PreparedForecastData) -> dict[str, object]:
    try:
        prediction_frame = get_pipeline().predict_df(
            prepared.context,
            prediction_length=prepared.horizon,
            quantile_levels=QUANTILES,
            id_column="item_id",
            timestamp_column="timestamp",
            target=prepared.target_columns,
            freq=prepared.frequency,
        )
    except Exception as exc:
        raise ForecastingError("No fue posible generar el pronóstico. Revisa la configuración y vuelve a intentarlo.") from exc

    predictions = [
        {
            "timestamp": row["timestamp"].isoformat(),
            "target": row["target_name"],
            "prediction": float(row["predictions"]),
            "lower": float(row["0.1"]),
            "median": float(row["0.5"]),
            "upper": float(row["0.9"]),
        }
        for _, row in prediction_frame.iterrows()
    ]
    return {
        "historical": prepared.historical,
        "forecast": predictions,
        "metadata": {
            "model": MODEL_ID,
            "horizon": prepared.horizon,
            "frequency": prepared.frequency,
            "target_columns": prepared.target_columns,
            "quantiles": QUANTILES,
        },
    }
