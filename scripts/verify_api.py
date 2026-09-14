"""Exercise the CSV upload and forecasting endpoints with Flask's test client."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app import app


def main() -> None:
    rows = ["timestamp,value"]
    rows.extend(f"2025-01-{day:02d},{120 + day}" for day in range(1, 31))
    csv_data = ("\n".join(rows) + "\n").encode()

    client = app.test_client()
    upload = client.post(
        "/api/upload",
        data={"file": (BytesIO(csv_data), "series.csv")},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201, upload.get_json()
    dataset_id = upload.get_json()["dataset_id"]

    try:
        response = client.post(
            "/api/forecast",
            json={
                "dataset_id": dataset_id,
                "timestamp_column": "timestamp",
                "target_columns": ["value"],
                "horizon": 4,
            },
        )
        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert len(body["forecast"]) == 4, body
        assert body["validation"]["train_observations"] == 26, body
        assert body["validation"]["validation_observations"] == 4, body
        assert {"mae", "rmse", "mape", "r2", "mase", "coverage"} <= set(body["validation"]["metrics"]["value"]), body
        assert body["metadata"]["covariate_columns"] == [], body
        print("API integration test: OK")
        print("Validation metrics:", body["validation"]["metrics"])
        print(body["forecast"])
    finally:
        Path(app.config["UPLOAD_FOLDER"], f"{dataset_id}.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
