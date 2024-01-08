from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from omegaconf import DictConfig
from prettytable import PrettyTable
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseModel(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        ...

    def save_model(self, save_dir: Path) -> None:
        joblib.dump(self.result, save_dir)

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> Any:
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def run_cv_training(self, X: pd.DataFrame, y: pd.Series) -> None:
        oof_preds = np.zeros((X.shape[0], y.unique().shape[0]))
        models = {}
        scores = PrettyTable()
        kfold = StratifiedKFold(n_splits=self.cfg.data.n_splits, shuffle=True, random_state=self.cfg.data.seed)

        for fold, (train_idx, valid_idx) in enumerate(iterable=kfold.split(X=X, y=y), start=1):
            with wandb.init(dir="never", project=self.cfg.experiment.project, name=f"cat-fold-{fold}"):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = self.fit(X_train, y_train, X_valid, y_valid)
                oof_preds[valid_idx] = (
                    model.predict(X_valid)
                    if isinstance(model, lgb.Booster)
                    else model.predict(xgb.DMatrix(X_valid))
                    if isinstance(model, xgb.Booster)
                    else model.predict_proba(X_valid.to_numpy()).reshape(-1)
                    if isinstance(model, TabNetClassifier)
                    else model.predict_proba(X_valid)
                )
                models[f"fold_{fold}"] = model
                scores.field_names = ["FOLD", "LogLoss"]
                scores.add_row([fold, log_loss(y_valid, oof_preds[valid_idx])])

            del X_train, X_valid, y_train, y_valid
            gc.collect()

        print(f"{scores.get_string()}")
        print(f"CV Score: {log_loss(y, oof_preds)}")

        self.result = ModelResult(oof_preds=oof_preds, models=models)
