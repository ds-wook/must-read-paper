# This code is a Python file that defines a function for preprocessing and loading datasets.
# The function reads in the dataset, performs necessary preprocessing tasks,
# and transforms it into a format that can be used for model training.
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from features.featurize import FeatureGenerator


def load_train_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(Path(cfg.data.path) / "train.csv")
    feature_generator = FeatureGenerator(cfg)
    train = feature_generator.generate_features(train)
    train = feature_generator.preprocess_train(train)
    X_train = train.drop(columns=[cfg.data.target])
    y_train = train[cfg.data.target]

    return X_train, y_train


def load_test_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    test = pd.read_csv(Path(cfg.data.path) / "test.csv")
    feature_generator = FeatureGenerator(cfg)
    test = feature_generator.generate_features(test)
    test = feature_generator.preprocess_test(test)

    return test
