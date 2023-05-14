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

# Contains code to preprocess data into input for the model
import json
import os
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from typing import Any, Dict, List

from constants import MAESTRO_DATA_PATH, PROCESSED_DATA_PATH
from midi import Midi
from embeddings.performance import Performance
from embeddings.melody import Melody
from embeddings.style import Style


def load_json(
    idx: int, encoding_type: str, data_path: str = PROCESSED_DATA_PATH
) -> Any:
    with open(os.path.join(data_path, encoding_type, f"{idx}.json"), "r") as f:
        return json.load(f)


def save_json(
    encoding: List[List[int]],
    idx: int,
    encoding_type: str,
    data_path: str = PROCESSED_DATA_PATH,
) -> None:
    with open(os.path.join(data_path, encoding_type, f"{idx}.json"), "w") as f:
        json.dump(encoding, f)


def load_midi(
    idx: int, data_path: str = PROCESSED_DATA_PATH
) -> Dict[str, List[List[int]]]:
    assert os.path.exists(
        os.path.join(data_path, "collated", f"{idx}.json")
    ), "No collated file found, please run collate in preprocess.py"

    return load_json(idx, "collated", data_path)


def load_combined(
    data_path: str = PROCESSED_DATA_PATH,
) -> Dict[str, Dict[str, List[List[int]]]]:
    assert os.path.exists(
        os.path.join(data_path, "combined.json")
    ), "No combined file found, please run combine in preprocess.py"

    with open(os.path.join(data_path, "combined.json"), "r") as f:
        return json.load(f)


def load_metadata(data_path: str = PROCESSED_DATA_PATH) -> Dict[str, Any]:
    assert os.path.exists(
        os.path.join(data_path, "metadata.json")
    ), "No metadata file found, please run summarize in preprocess.py"

    with open(os.path.join(data_path, "metadata.json"), "r") as f:
        return json.load(f)


def _maestro_midi_generator(index, pbar):
    for idx in index:
        if pbar is not None:
            pbar.update()

        try:
            yield Midi.get_sample_from_maestro(idx)
        except AssertionError:
            # Skipping files that don't meet the requirements
            print(f"Skipping {idx}")


def compress_encoding(encoding: np.ndarray) -> List[List[int]]:
    # Expects start and end tokens to be added
    compressed = []
    for row in encoding:
        encoding = sum(np.argwhere(row == 1).tolist(), [])

        # Forward fill missing values
        encoding = encoding if len(encoding) > 0 else compressed[-1]
        compressed.append(encoding)

    return compressed


def expand_encoding(
    compressed: List[List[int]], start: int = 0, end: int = -1
) -> np.ndarray:
    assert start >= 0 and start < len(compressed), "Invalid start index"
    assert end == -1 or (end > start and end <= len(compressed)), "Invalid end index"
    end = len(compressed) if end == -1 else end

    # Every encoding contains start token
    expanded = np.zeros((end - start, compressed[0][0] + 2), dtype=np.float32)
    compressed = compressed[start:end]

    for encoding, row in zip(expanded, compressed):
        encoding[row] = 1

    return expanded


def _add_start_and_end_tokens(data: np.ndarray) -> np.ndarray:
    data = np.vstack((np.zeros((data.shape[1])), data, np.zeros((data.shape[1]))))
    data = np.hstack((data, np.zeros((data.shape[0], 2))))
    data[0, -2] = 1
    data[-1, -1] = 1
    return data


def inflate(
    compressed: List[List[int]], perf: Performance, midi: Midi
) -> List[List[int]]:
    # Inflates melody and style encoding to line up temporally with performance encoding
    # Expects start and end tokens to be added to compressed encoding
    return [
        compressed[np.clip(midi.time_to_tick(exact_time), 0, len(compressed) - 1)]
        for exact_time in np.concatenate([[0], perf.exact_times, [1e10]])
    ]


def process_midi(
    midi: Midi,
    out_path: str = PROCESSED_DATA_PATH,
    use_cache: bool = True,
    Encoders=(Melody, Style, Performance),
) -> None:
    if midi is None:
        return

    if use_cache and all(
        os.path.exists(
            os.path.join(out_path, Encoder.encoding_type, f"{midi.idx}.json")
        )
        for Encoder in Encoders
    ):
        return

    if Melody in Encoders or Style in Encoders:
        assert (
            Performance in Encoders
        ), "Performance encoding is required for melody and style encoding"

    encodings = {Encoder.encoding_type: Encoder(midi) for Encoder in Encoders}
    performance = encodings[Performance.encoding_type]

    encodings = [_add_start_and_end_tokens(encoding.encoding) for encoding in encodings]
    encodings = [compress_encoding(encoding) for encoding in encodings]
    encodings[:2] = [inflate(encoding, performance, midi) for encoding in encodings[:2]]

    [
        save_json(encoding, midi.idx, Encoder.encoding_type, out_path)
        for encoding, Encoder in zip(encodings, Encoders)
    ]


def summarize(data_path: str = PROCESSED_DATA_PATH) -> None:
    # Creates metadata file containing information about the dataset
    metadata = {}
    files = os.listdir(os.path.join(data_path, Performance.encoding_type))

    metadata["num_songs"] = len(files)
    metadata["num_samples"] = {}

    total_samples = 0
    for i, file in tqdm(enumerate(files), total=len(files), desc="Summarizing"):
        idx = int(file.split(".")[0])
        embedding = load_json(idx, Performance.encoding_type, data_path)
        metadata["num_samples"][idx] = len(embedding)
        total_samples += len(embedding)

        if i == 0:
            # Start token is at index 2 less than dimension of embedding
            metadata["d_embd_performance"] = embedding[0][0] + 2
            metadata["d_embd_melody"] = load_json(idx, "melody")[0][0] + 2
            metadata["d_embd_style"] = load_json(idx, "style")[0][0] + 2

    metadata["total_samples"] = total_samples

    with open(os.path.join(data_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def collate(data_path: str = PROCESSED_DATA_PATH) -> None:
    # Combines performance, melody, and style encodings into a single file for more efficient loading
    files = os.listdir(os.path.join(data_path, Performance.encoding_type))

    for file in tqdm(files, desc="Collating"):
        idx = int(file.split(".")[0])
        data = {
            Performance.encoding_type: load_json(
                idx, Performance.encoding_type, data_path
            ),
            Melody.encoding_type: load_json(idx, Melody.encoding_type, data_path),
            Style.encoding_type: load_json(idx, Style.encoding_type, data_path),
        }

        save_json(data, idx, "collated", data_path)


def combine(data_path: str = PROCESSED_DATA_PATH) -> None:
    if not os.path.exists(os.path.join(data_path, "collated")):
        collate(data_path)

    # To improve data loading, store all data compressed in a single file
    # May require more memory

    files = os.listdir(os.path.join(data_path, "collated"))
    data = {}
    for file in tqdm(files, desc="Combining"):
        idx = file.split(".")[0]
        data[idx] = load_json(idx, "collated", data_path)

    with open(os.path.join(data_path, "combined.json"), "w") as f:
        output = json.dumps(data)
        f.write(output)


def uncombine(data_path: str = PROCESSED_DATA_PATH):
    print("Loading Full Dataset")
    data = load_combined(data_path)
    for idx, data in tqdm(data.items(), desc="Uncombining"):
        save_json(data, idx, "collated", data_path)


def process_all(
    out_path: str = PROCESSED_DATA_PATH,
    use_cache: bool = True,
    parallel: bool = True,
    n_jobs: int = 6,
) -> None:
    maestro = pd.read_csv(MAESTRO_DATA_PATH)
    N = len(maestro)

    with tqdm(total=N, desc="Processing") as pbar:
        midis = _maestro_midi_generator(maestro.index, pbar)
        process = partial(process_midi, out_path=out_path, use_cache=use_cache)

        if parallel:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                executor.map(process, midis)
        else:
            for midi in midis:
                process(midi)

    summarize(out_path)
    collate(out_path)
    combine(out_path)


if __name__ == "__main__":
    # collate()
    uncombine()
    # summarize()
