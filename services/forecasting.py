"""Thin adapter around the installed Chronos-2 forecasting API."""

from __future__ import annotations

import numpy as np

from services.chronos_model import MODEL_ID, QUANTILES, predict
from services.data_processing import PreparedForecastData


class ForecastingError(RuntimeError):
    """An inference failure that should not expose internal details to clients."""


def _predict(context, prepared: PreparedForecastData, *, future_df):
    """Run one Chronos-2 pass; extra columns in the context act as past covariates."""
    return predict(
        context,
        future_df=future_df,
        horizon=prepared.horizon,
        target_columns=prepared.target_columns,
        frequency=prepared.frequency,
    )


def forecast(prepared: PreparedForecastData) -> dict[str, object]:
    try:
        train_frame = prepared.context.iloc[: -prepared.horizon].copy()
        validation_frame = prepared.context.iloc[-prepared.horizon :].copy()

        # During validation the holdout covariates are already observed, so they are passed
        # as known future covariates. Beyond the end of the CSV they are unknown, so the
        # future forecast can only use them as past covariates.
        future_covariates = (
            validation_frame[["item_id", "timestamp", *prepared.covariate_columns]]
            if prepared.covariate_columns
            else None
        )
        validation_prediction_frame = _predict(train_frame, prepared, future_df=future_covariates)
        prediction_frame = _predict(prepared.context, prepared, future_df=None)
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
            "covariate_columns": prepared.covariate_columns,
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

        # Raw errors are unreadable on their own, so every run is scored against the two
        # baselines it must beat to be worth anything: repeating the last observation
        # (MASE) and predicting the training mean (R2).
        train_values = prepared.context[target].to_numpy(dtype=float)[: -prepared.horizon]
        naive_mae = float(np.mean(np.abs(actual_values - train_values[-1])))
        variance = float(np.sum(np.square(actual_values - actual_values.mean())))
        inside = (actual_values >= comparison["0.1"].to_numpy(dtype=float)) & (
            actual_values <= comparison["0.9"].to_numpy(dtype=float)
        )

        metrics[target] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mape": float(np.mean(np.abs(errors[nonzero] / actual_values[nonzero])) * 100) if np.any(nonzero) else None,
            "mape_excluded": int((~nonzero).sum()),
            "r2": float(1 - np.sum(np.square(errors)) / variance) if variance > 0 else None,
            "mase": float(np.mean(np.abs(errors)) / naive_mae) if naive_mae > 0 else None,
            "coverage": float(np.mean(inside) * 100),
            "nominal_coverage": (QUANTILES[-1] - QUANTILES[0]) * 100,
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
