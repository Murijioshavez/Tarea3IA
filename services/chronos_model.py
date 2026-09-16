"""Dedicated access layer for Amazon Chronos-2.

This module contains no training code. It loads the official pre-trained
``amazon/chronos-2`` checkpoint once and exposes its supported DataFrame
inference API to the rest of the application.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))

from chronos import Chronos2Pipeline


MODEL_ID = "amazon/chronos-2"
QUANTILES = [0.1, 0.5, 0.9]
_pipeline: Chronos2Pipeline | None = None
_pipeline_lock = Lock()


def get_model() -> Chronos2Pipeline:
    """Return the cached, pre-trained Chronos-2 pipeline running on CPU."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")
    return _pipeline


def predict(context, *, target_columns, horizon: int, frequency: str, future_df=None):
    """Forecast a prepared DataFrame using Chronos-2; no fitting is performed."""
    return get_model().predict_df(
        context,
        future_df=future_df,
        prediction_length=horizon,
        quantile_levels=QUANTILES,
        id_column="item_id",
        timestamp_column="timestamp",
        target=target_columns,
        freq=frequency,
    )
