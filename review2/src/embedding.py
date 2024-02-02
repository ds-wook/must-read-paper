from __future__ import annotations

import gc
import os
import warnings
from pathlib import Path

import esm
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm


def save_embedding(data: list[tuple[int, str]], file_name: Path, batch_size: int = 8) -> None:
    torch.cuda.empty_cache()
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.cuda()

    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    print("data length:", len(data))
    print("batches num:", len(batches))

    sequence_representations = []

    for batch in tqdm(batches):
        _, _, batch_tokens = batch_converter(batch)

        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        for i, (_, seq) in enumerate(batch):
            sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))

        del batch_tokens
        gc.collect()
        torch.cuda.empty_cache()

    sequence_representations = np.array([emb.detach().cpu().numpy() for emb in sequence_representations])
    print("embedding shape:", sequence_representations.shape)

    try:
        np.save(file_name, sequence_representations)
    except FileNotFoundError:
        os.mkdir("resources/encoder/")
        np.save(file_name, sequence_representations)


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(cfg.data.encoder)
        df_train = pd.read_csv(Path(cfg.data.path) / "train.csv", index_col="id")
        df_test = pd.read_csv(Path(cfg.data.path) / "test.csv", index_col="id")

        window_size = 256

        train_antigen_seq_left = [
            seq[int(start_pos) - 1 - window_size : int(start_pos) - 1]
            for seq, start_pos in zip(df_train.antigen_seq, df_train.start_position)
        ]
        test_antigen_seq_left = [
            seq[int(start_pos) - 1 - window_size : int(start_pos) - 1]
            for seq, start_pos in zip(df_test.antigen_seq, df_test.start_position)
        ]

        train_antigen_seq_right = [
            seq[int(end_pos) : window_size + int(end_pos)]
            for seq, end_pos in zip(df_train.antigen_seq, df_train.end_position)
        ]
        test_antigen_seq_right = [
            seq[int(end_pos) : window_size + int(end_pos)]
            for seq, end_pos in zip(df_test.antigen_seq, df_test.end_position)
        ]

        save_embedding(list(zip(df_train.index, df_train.epitope_seq)), save_path / "temp_train_epitope_esm_emb.npy")
        save_embedding(
            list(zip(df_train.index, train_antigen_seq_left)), save_path / "temp_train_antigen_left_esm_emb.npy"
        )
        save_embedding(
            list(zip(df_train.index, train_antigen_seq_right)), save_path / "temp_train_antigen_right_esm_emb.npy"
        )

        save_embedding(list(zip(df_test.index, df_test.epitope_seq)), save_path / "temp_test_epitope_esm_emb.npy")
        save_embedding(
            list(zip(df_test.index, test_antigen_seq_left)), save_path / "temp_test_antigen_left_esm_emb.npy"
        )
        save_embedding(
            list(zip(df_test.index, test_antigen_seq_right)), save_path / "temp_test_antigen_right_esm_emb.npy"
        )


if __name__ == "__main__":
    _main()
