from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - fastapi is optional
    FastAPI = None

try:
    import joblib
except ImportError:  # pragma: no cover - joblib is required for serving
    joblib = None


class Serve:
    """Serve the best model via FastAPI."""

    def __init__(
        self, endpoint_name: str = "/model/predict", model_dir: str = "models"
    ) -> None:
        if not endpoint_name.startswith("/"):
            endpoint_name = f"/{endpoint_name}"
        self.endpoint_name = endpoint_name
        self.model_dir = Path(model_dir)
        self.best_model_file = self.model_dir / "best_model.joblib"
        self.best_metric_file = self.model_dir / "best_model.json"

    def _load_model(self):
        if joblib is None:
            raise ImportError("joblib is required for Serve step")
        if not self.best_model_file.exists():
            raise FileNotFoundError("Best model not found. Run the Model step first.")
        return joblib.load(self.best_model_file)

    def create_app(self) -> Any:
        if FastAPI is None:
            raise ImportError("fastapi is required for Serve step")
        model = self._load_model()
        metrics: Dict[str, Any] = {}
        if self.best_metric_file.exists():
            try:
                with open(self.best_metric_file) as f:
                    metrics = json.load(f)
            except json.JSONDecodeError:
                metrics = {}

        app = FastAPI()

        @app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}

        @app.get("/metrics")
        def get_metrics() -> Dict[str, Any]:
            return metrics

        @app.post(self.endpoint_name)
        def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
            df = pd.DataFrame([payload])
            preds: List[Any] = model.predict(df).tolist()
            return {"prediction": preds[0] if len(preds) == 1 else preds}

        return app

    def run(self, data: Any = None) -> Any:
        """Return a FastAPI app instance."""
        return self.create_app()
