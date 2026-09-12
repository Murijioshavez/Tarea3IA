"""Run a minimal, standalone Chronos-2 inference on CPU.

This script deliberately remains outside Flask. It validates the installed
runtime, downloads the official model if necessary, and prints the DataFrame
returned by the supported Chronos-2 pandas API.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import pandas as pd
from chronos import Chronos2Pipeline


def build_context() -> pd.DataFrame:
    """Create one regular daily target series for the smoke test."""
    return pd.DataFrame(
        {
            "item_id": "smoke-test-series",
            "timestamp": pd.date_range("2025-01-01", periods=30, freq="D"),
            "target": [
                120,
                123,
                121,
                125,
                129,
                127,
                132,
                136,
                133,
                139,
                141,
                138,
                144,
                147,
                145,
                150,
                153,
                151,
                156,
                159,
                157,
                162,
                165,
                163,
                168,
                171,
                169,
                174,
                177,
                175,
            ],
        }
    )


def main() -> None:
    context_df = build_context()
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
    )
    forecast_df = pipeline.predict_df(
        context_df,
        prediction_length=4,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    )

    print("Input rows:", len(context_df))
    print("Forecast rows:", len(forecast_df))
    print(forecast_df.to_string(index=False))


if __name__ == "__main__":
    main()
