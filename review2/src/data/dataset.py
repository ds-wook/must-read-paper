from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from data.preprocessing import LabelEncoder


def categorize_train_features(cfg: DictConfig, train: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        config: config
        train: dataframe
    Returns:
        dataframe
    """

    path = Path(cfg.data.encoder)

    le = LabelEncoder(min_obs=10)
    train[[*cfg.data.cat_features]] = le.fit_transform(train[[*cfg.data.cat_features]])
    joblib.dump(le, path / "label_encoder.pkl")

    return train


def categorize_test_features(cfg: DictConfig, test: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        config: config
        test: dataframe
    Returns:
        dataframe
    """

    path = Path(cfg.data.encoder)

    le = joblib.load(path / "label_encoder.pkl")
    test[[*cfg.data.cat_features]] = le.transform(test[[*cfg.data.cat_features]])

    return test


def input_preprocessing(
    df: pd.DataFrame, selected_columns: list[str], embeddings: list[tuple[np.array, str]]
) -> pd.DataFrame:
    df_input = [df[selected_columns].copy()]

    for embedding, prefix in tqdm(embeddings, leave=False):
        emb_dim = len(embedding[0])
        df_input.append(pd.DataFrame(embedding, columns=[f"{prefix}{i}" for i in range(emb_dim)]))

    df_input = pd.concat(df_input, axis=1)

    return df_input


def load_train_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    train_epitope_esm_emb = np.load(Path(cfg.data.encoder) / "temp_train_epitope_esm_emb.npy")
    train_antigen_left_esm_emb = np.load(Path(cfg.data.encoder) / "temp_train_antigen_left_esm_emb.npy")
    train_antigen_right_esm_emb = np.load(Path(cfg.data.encoder) / "temp_train_antigen_right_esm_emb.npy")

    train = pd.read_csv(Path(cfg.data.path) / "train.csv").drop(columns=["id"])

    train_x = input_preprocessing(
        train,
        [*cfg.data.input_features],
        [
            (train_epitope_esm_emb, "epitope_esm_emb_"),
            (train_antigen_left_esm_emb, "antigen_left_esm_emb_"),
            (train_antigen_right_esm_emb, "antigen_right_esm_emb_"),
        ],
    )
    train_x = categorize_train_features(cfg, train_x)

    train_y = train[cfg.data.target]
    train_x[[*cfg.data.cat_features]] = train_x[[*cfg.data.cat_features]].astype(int)

    return train_x, train_y


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    test_epitope_esm_emb = np.load(Path(cfg.data.encoder) / "temp_test_epitope_esm_emb.npy")
    test_antigen_left_esm_emb = np.load(Path(cfg.data.encoder) / "temp_test_antigen_left_esm_emb.npy")
    test_antigen_right_esm_emb = np.load(Path(cfg.data.encoder) / "temp_test_antigen_right_esm_emb.npy")

    test = pd.read_csv(Path(cfg.data.path) / "test.csv").drop(columns=["id"])

    test_x = input_preprocessing(
        test,
        cfg.data.input_features,
        [
            (test_epitope_esm_emb, "epitope_esm_emb_"),
            (test_antigen_left_esm_emb, "antigen_left_esm_emb_"),
            (test_antigen_right_esm_emb, "antigen_right_esm_emb_"),
        ],
    )
    test_x = categorize_test_features(cfg, test_x)
    test_x[[*cfg.data.cat_features]] = test_x[[*cfg.data.cat_features]].astype(int)

    return test_x
