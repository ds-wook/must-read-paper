from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from features.base import BasePreprocessor


class FeatureGenerator(BasePreprocessor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        # absolute
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
        df["DAYS_EMPLOYED"] = np.abs(df["DAYS_EMPLOYED"])
        df["DAYS_BIRTH"] = np.abs(df["DAYS_BIRTH"])
        df["begin_month"] = np.abs(df["begin_month"]).astype(int)

        # DAYS_BIRTH
        df["DAYS_BIRTH_MONTH"] = np.floor(df["DAYS_BIRTH"] / 30) - (
            (np.floor(df["DAYS_BIRTH"] / 30) / 12).astype(int) * 12
        )
        df["DAYS_BIRTH_MONTH"] = df["DAYS_BIRTH_MONTH"].astype(int)
        df["DAYS_BIRTH_WEEK"] = np.floor(df["DAYS_BIRTH"] / 7) - ((np.floor(df["DAYS_BIRTH"] / 7) / 4).astype(int) * 4)
        df["DAYS_BIRTH_WEEK"] = df["DAYS_BIRTH_WEEK"].astype(int)

        # Age
        df["Age"] = np.abs(df["DAYS_BIRTH"]) // 360

        # DAYS_EMPLOYED
        df["DAYS_EMPLOYED_MONTH"] = np.floor(df["DAYS_EMPLOYED"] / 30) - (
            (np.floor(df["DAYS_EMPLOYED"] / 30) / 12).astype(int) * 12
        )
        df["DAYS_EMPLOYED_MONTH"] = df["DAYS_EMPLOYED_MONTH"].astype(int)
        df["DAYS_EMPLOYED_WEEK"] = np.floor(df["DAYS_EMPLOYED"] / 7) - (
            (np.floor(df["DAYS_EMPLOYED"] / 7) / 4).astype(int) * 4
        )
        df["DAYS_EMPLOYED_WEEK"] = df["DAYS_EMPLOYED_WEEK"].astype(int)

        # EMPLOYED
        df["EMPLOYED"] = df["DAYS_EMPLOYED"] / 360

        # before_EMPLOYED
        df["BEFORE_EMPLOYED"] = df["DAYS_BIRTH"] - df["DAYS_EMPLOYED"]
        df["BEFORE_EMPLOYED_MONTH"] = np.floor(df["BEFORE_EMPLOYED"] / 30) - (
            (np.floor(df["BEFORE_EMPLOYED"] / 30) / 12).astype(int) * 12
        )
        df["BEFORE_EMPLOYED_MONTH"] = df["BEFORE_EMPLOYED_MONTH"].astype(int)
        df["BEFORE_EMPLOYED_WEEK"] = np.floor(df["BEFORE_EMPLOYED"] / 7) - (
            (np.floor(df["BEFORE_EMPLOYED"] / 7) / 4).astype(int) * 4
        )
        df["BEFORE_EMPLOYED_WEEK"] = df["BEFORE_EMPLOYED_WEEK"].astype(int)

        # user code
        df["USER"] = df["gender"].astype(str) + "_" + df["car"].astype(str) + "_" + df["reality"].astype(str)

        return df

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        df = self._add_features(df)

        return df
