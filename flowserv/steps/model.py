from __future__ import annotations

from typing import Any, Optional

from pathlib import Path

import json
import os
import time

import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars may not be installed
    pl = None

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

from joblib import dump

ALLOWED_OPTIMIZERS = {"accuracy", "f1", "r2", "rmse"}


class Model:
    """Train a simple ML model using scikit-learn."""

    def __init__(
        self,
        target: str,
        infer_model: bool = True,
        model_type: str = "infer",
        model: Optional[Any] = None,
        track_experiments: bool = False,
        optimizer: str = "accuracy",
        model_dir: str = "models",
    ) -> None:
        self.target = target
        self.infer_model = infer_model
        self.model_type = model_type
        self.model = model
        self.track_experiments = track_experiments
        if optimizer not in ALLOWED_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {ALLOWED_OPTIMIZERS}")
        self.optimizer = optimizer
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_file = self.model_dir / "best_model.joblib"
        self.best_metric_file = self.model_dir / "best_model.json"
        self.metrics = {}

    def _infer_task(self, y: pd.Series) -> str:
        return "regression" if pd.api.types.is_numeric_dtype(y) else "classification"

    def _default_model(self, task: str):
        return (
            RandomForestClassifier()
            if task == "classification"
            else RandomForestRegressor()
        )

    def run(self, data: Any):
        if pl is not None and isinstance(data, pl.LazyFrame):
            df = data.collect().to_pandas()
        elif pl is not None and isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(
                "Model step expects a polars LazyFrame/DataFrame or pandas DataFrame"
            )

        X = df.drop(columns=[self.target])
        y = df[self.target]

        task = self.model_type if self.model_type != "infer" else self._infer_task(y)
        model = (
            self.model
            if not self.infer_model and self.model is not None
            else self._default_model(task)
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "classification":
            self.metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds, average="weighted"),
            }
        else:
            self.metrics = {
                "r2": r2_score(y_test, preds),
                "rmse": mean_squared_error(y_test, preds, squared=False),
            }

        score = self.metrics.get(self.optimizer)

        if self.track_experiments and wandb is not None:
            run = wandb.init(
                project="flowserve",
                reinit=True,
                mode=os.getenv("WANDB_MODE", "offline"),
            )
            wandb.log(self.metrics)
            tmp_model_path = self.model_dir / f"model_{int(time.time())}.joblib"
            dump(model, tmp_model_path)
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(str(tmp_model_path))
            wandb.log_artifact(artifact)
            run.finish()
        else:
            tmp_model_path = self.model_dir / f"model_{int(time.time())}.joblib"
            dump(model, tmp_model_path)

        best_score = -float("inf")
        if self.best_metric_file.exists():
            with open(self.best_metric_file) as f:
                try:
                    data = json.load(f)
                    best_score = data.get("score", best_score)
                except json.JSONDecodeError:
                    pass
        if score is not None and score > best_score:
            dump(model, self.best_model_file)
            with open(self.best_metric_file, "w") as f:
                json.dump({"score": score, "metric": self.optimizer}, f)

        self.model = model
        return model
