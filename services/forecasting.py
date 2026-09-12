"""Thin adapter around the installed Chronos-2 forecasting API."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import numpy as np

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
        train_frame = prepared.context.iloc[: -prepared.horizon].copy()
        validation_frame = prepared.context.iloc[-prepared.horizon :].copy()
        validation_prediction_frame = get_pipeline().predict_df(
            train_frame,
            prediction_length=prepared.horizon,
            quantile_levels=QUANTILES,
            id_column="item_id",
            timestamp_column="timestamp",
            target=prepared.target_columns,
            freq=prepared.frequency,
        )
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
        "validation": build_validation_result(validation_frame, validation_prediction_frame, prepared),
        "metadata": {
            "model": MODEL_ID,
            "horizon": prepared.horizon,
            "frequency": prepared.frequency,
            "target_columns": prepared.target_columns,
            "quantiles": QUANTILES,
        },
    }


def build_validation_result(validation_frame, prediction_frame, prepared: PreparedForecastData) -> dict[str, object]:
    """Compare Chronos-2 predictions against a holdout unseen during inference."""
    predictions: list[dict[str, object]] = []
    metrics: dict[str, dict[str, float | None]] = {}
    for target in prepared.target_columns:
        actual = validation_frame[["timestamp", target]].rename(columns={target: "actual"})
        estimated = prediction_frame[prediction_frame["target_name"] == target]
        comparison = actual.merge(estimated, on="timestamp", how="inner")
        if len(comparison) != prepared.horizon:
            raise ForecastingError("La validación no produjo el número esperado de predicciones.")
        actual_values = comparison["actual"].to_numpy(dtype=float)
        forecast_values = comparison["predictions"].to_numpy(dtype=float)
        errors = actual_values - forecast_values
        nonzero = actual_values != 0
        metrics[target] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mape": float(np.mean(np.abs(errors[nonzero] / actual_values[nonzero])) * 100) if np.any(nonzero) else None,
        }
        predictions.extend(
            {
                "timestamp": row["timestamp"].isoformat(),
                "target": target,
                "actual": float(row["actual"]),
                "prediction": float(row["predictions"]),
                "lower": float(row["0.1"]),
                "upper": float(row["0.9"]),
            }
            for _, row in comparison.iterrows()
        )
    return {
        "train_observations": len(prepared.context) - prepared.horizon,
        "validation_observations": prepared.horizon,
        "predictions": predictions,
        "metrics": metrics,
    }
