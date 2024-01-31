from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig, OmegaConf
from scipy.misc import derivative
from sklearn.metrics import f1_score

from models.base import BaseModel


class XGBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        model = xgb.train(
            OmegaConf.to_container(self.cfg.models.params),
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            num_boost_round=self.cfg.models.num_boost_round,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
            verbose_eval=self.cfg.models.verbose_eval,
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> CatBoostClassifier:
        train_set = Pool(X_train, y_train, cat_features=self.cfg.features.cat_features)
        valid_set = Pool(X_valid, y_valid, cat_features=self.cfg.features.cat_features)

        model = CatBoostClassifier(
            random_state=self.cfg.models.seed,
            cat_features=self.cfg.features.cat_features,
            **self.cfg.models.params,
        )

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.cfg.models.verbose_eval,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
        )

        return model


class LightGBMTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=self.cfg.data.cat_features)
        valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=self.cfg.data.cat_features)

        model = lgb.train(
            # fobj=lambda y_hat, data: self._focal_loss_lgb(y_hat, data, 0.25, 2.0),
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            feval=self._f1_score,
            params=OmegaConf.to_container(self.cfg.models.params),
            num_boost_round=self.cfg.models.num_boost_round,
            callbacks=[
                lgb.log_evaluation(self.cfg.models.verbose_eval),
                lgb.early_stopping(self.cfg.models.early_stopping_rounds),
            ],
        )

        return model

    def _f1_score(self, y_hat: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
        y_true = data.get_label()
        y_hat = np.where(y_hat < 0.5, 0, 1)
        return "f1", f1_score(y_true, y_hat, average="macro"), True

    def _focal_loss_lgb(self, y_pred, dtrain, alpha, gamma):
        """
        Focal Loss for lightgbm

        Parameters:
        -----------
        y_pred: numpy.ndarray
            array with the predictions
        dtrain: lightgbm.Dataset
        alpha, gamma: float
            See original paper https://arxiv.org/pdf/1708.02002.pdf
        """
        a, g = alpha, gamma
        y_true = dtrain.label

        def fl(x, t):
            p = 1 / (1 + np.exp(-x))
            return (
                -(a * t + (1 - a) * (1 - t))
                * ((1 - (t * p + (1 - t) * (1 - p))) ** g)
                * (t * np.log(p) + (1 - t) * np.log(1 - p))
            )

        grad = derivative(lambda x: fl(x, y_true), y_pred, n=1, dx=1e-6)
        hess = derivative(lambda x: fl(x, y_true), y_pred, n=2, dx=1e-6)
        return grad, hess
