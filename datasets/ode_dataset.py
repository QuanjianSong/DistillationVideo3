import os
from datasets.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }