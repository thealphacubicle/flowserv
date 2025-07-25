from __future__ import annotations

from typing import Any, Optional

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
from sklearn.metrics import accuracy_score, r2_score

class Model:
    """Train a simple ML model using scikit-learn."""

    def __init__(
        self,
        target: str,
        infer_model: bool = True,
        model_type: str = "infer",
        model: Optional[Any] = None,
        track_experiments: bool = False,
    ) -> None:
        self.target = target
        self.infer_model = infer_model
        self.model_type = model_type
        self.model = model
        self.track_experiments = track_experiments
        self.metrics = {}

    def _infer_task(self, y: pd.Series) -> str:
        return "regression" if pd.api.types.is_numeric_dtype(y) else "classification"

    def _default_model(self, task: str):
        return RandomForestClassifier() if task == "classification" else RandomForestRegressor()

    def run(self, data: Any):
        if pl is not None and isinstance(data, pl.LazyFrame):
            df = data.collect().to_pandas()
        elif pl is not None and isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Model step expects a polars LazyFrame/DataFrame or pandas DataFrame")

        X = df.drop(columns=[self.target])
        y = df[self.target]

        task = self.model_type if self.model_type != "infer" else self._infer_task(y)
        model = self.model if not self.infer_model and self.model is not None else self._default_model(task)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "classification":
            self.metrics = {"accuracy": accuracy_score(y_test, preds)}
        else:
            self.metrics = {"r2": r2_score(y_test, preds)}

        if self.track_experiments and wandb is not None:
            wandb.log(self.metrics)

        self.model = model
        return model
