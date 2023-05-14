# Copyright 2023 Vikaas Varma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from constants import DEVICE, PROCESSED_DATA_PATH, LOAD_FULL_DATASET, MAESTRO_DATA_PATH
from hyperparameters import MINI_BATCH_SIZE, MAX_SEQ_LEN
from preprocess import load_metadata, load_midi, load_combined, expand_encoding

np.random.seed(5318008)


class MIDIEmbeddings(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        perturb_performance: bool = True,
        device=DEVICE,
        load_full_dataset=True,
        window_size=MAX_SEQ_LEN // 4,
    ):
        self.perturb_performance = perturb_performance
        self.device = device
        self.load_full_dataset = load_full_dataset
        self.metadata = load_metadata()
        self.window_size = window_size
        indices = set(indices.astype(str))

        # Precalculate index of each sample in the dataset
        self.num_samples = np.array(
            [
                (int(idx), count)
                for idx, count in self.metadata["num_samples"].items()
                if idx in indices
            ]
        )
        self.num_samples = self.num_samples[self.num_samples[:, 0].argsort()]

        self.start_idx = np.random.randint(window_size)
        self.init_locations()
        self.cache = OrderedDict()
        self.cache_size = 4

        if load_full_dataset:
            print("Loading Dataset into Memory...")
            self.dataset = load_combined(PROCESSED_DATA_PATH)

    def init_locations(self):
        # Cumulative sum of number of data points in each song
        self.indices = self.num_samples[:, 0].astype(str)

        # Number of time steps in each song starting from start_idx
        steps_per_song = self.num_samples[:, 1] - self.start_idx - MAX_SEQ_LEN

        # Number of time steps when jumping window_size at a time
        windows_per_song = np.ceil((steps_per_song - 1) / self.window_size) + 1
        windows_per_song += 1 if self.start_idx != 0 else 0

        # Global index of starting sample of each song
        self.locations = np.cumsum(windows_per_song.astype(int))
        self.length = self.locations[-1]

    def shuffle(self):
        self.start_idx = np.random.randint(self.window_size)
        np.random.shuffle(self.num_samples)
        self.init_locations()

    def load_idx(self, song_idx: int) -> Dict[List[List[int]]]:
        if self.load_full_dataset:
            return self.dataset[self.indices[song_idx]]

        if (key := self.indices[song_idx]) in self.cache:
            return self.cache[key]

        song = load_midi(key, PROCESSED_DATA_PATH)

        self.cache[key] = {
            encoding_type: expand_encoding(encoding)
            for encoding_type, encoding in song.items()
        }

        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return self.cache[key]

    def perturb(self, p: np.ndarray) -> np.ndarray:
        # Perturb performance by randomly shifting pitch and time
        dp = np.random.choice([1, 2, 3, 4, 5, 6]) * np.random.choice([-1, 1])
        dt = 1 + np.random.choice([2, 4]) * np.random.choice([-1, 1])

        # Shift note on and note off events while respecting embedding bounds
        # Note zeroing edges is unnecessary for pitches since piano lies between 21 and 108
        p[:, max(dp, 0) : min(256 + dp, 256)] = p[:, max(-dp, 0) : min(256 - dp, 256)]

        p[:, max(dt, 0) + 256 : min(125 + dt, 125) + 256] = p[
            :, max(-dt, 0) + 256 : min(125 - dt, 125) + 256
        ]
        p[:, 256 : max(dt, 0) + 256] = 0
        p[:, min(125 + dt, 125) + 256 : 256 + 125] = 0

        return p

    def __len__(self):
        return self.length

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Model is autoencoder, so output is the same as input performance embedding

        # Gets index of idx if sorted in self.locations (id of song)
        song_idx = np.searchsorted(self.locations, idx, side="right")
        data = self.load_idx(song_idx)

        # Get index of beginning of sequence in song
        index = (idx - 1 - (0 if song_idx == 0 else self.locations[song_idx - 1])) * (
            self.window_size
        ) + self.start_idx
        index = np.clip(index, 0, len(data["melody"]) - MAX_SEQ_LEN - 1)

        melody = data["melody"][index + 1 : index + MAX_SEQ_LEN + 1]
        style = data["style"][index + 1 : index + MAX_SEQ_LEN + 1]
        conditioning_perf = data["performance"][index + 1 : index + MAX_SEQ_LEN + 1]
        previous_perf = data["performance"][index : index + MAX_SEQ_LEN]

        # On training, perturb performance
        if self.perturb_performance:
            conditioning_perf = self.perturb(conditioning_perf)

        return (
            melody,
            style,
            conditioning_perf,
            previous_perf,
        )


def train_test_val_split(
    device: str = DEVICE,
    load_full_dataset: bool = LOAD_FULL_DATASET,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    maestro = pd.read_csv(MAESTRO_DATA_PATH)

    # Shuffling is performed in dataset to improve performance
    train, val, test = (
        MIDIEmbeddings(
            maestro[maestro["split"] == split].index,
            device=device,
            load_full_dataset=load_full_dataset,
        )
        for split in ("train", "validation", "test")
    )

    # Perform batch collating in numpy to improve performance
    def collate_fn(tensors):
        melodies, styles, conditionings, previous = (
            torch.from_numpy(np.array(tensor)).to(device) for tensor in zip(*tensors)
        )
        return ((melodies, styles, conditionings, previous), conditionings)

    return (
        DataLoader(
            data, batch_size=MINI_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )
        for data in (train, val, test)
    )


if __name__ == "__main__":
    x, _, _ = train_test_val_split()
    x.dataset.shuffle()
    for i, profile in tqdm(enumerate(x), total=len(x)):
        pass
