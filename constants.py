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

import numpy as np
import torch

MAESTRO_DATA_PATH = "data/maestro-v3.0.0/maestro-v3.0.0.csv"
CHORD_ENCODINGS_PATH = "data/chord_encoding.json"
VAMP_PATH = "/Users/vikaas/Library/Audio/Plug-Ins/Vamp"
PROCESSED_DATA_PATH = "data/processed"
MODEL_PATH = "/rds/user/pt442/hpc-work/vik"
LOAD_FULL_DATASET = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Average Equal loudness Response Curve
# https://replaygain.hydrogenaud.io/equal_loudness.html
AVERAGE_EQUAL_LOUDNESS_CURVE = np.array(
    [
        [20, 113],
        [30, 103],
        [40, 97],
        [50, 93],
        [60, 91],
        [70, 89],
        [80, 87],
        [90, 86],
        [100, 85],
        [200, 78],
        [300, 76],
        [400, 76],
        [500, 76],
        [600, 76],
        [700, 77],
        [800, 78],
        [900, 79.5],
        [1000, 80],
        [1500, 79],
        [2000, 77],
        [2500, 74],
        [3000, 71.5],
        [3700, 70],
        [4000, 70.5],
        [5000, 74],
        [6000, 79],
        [7000, 84],
        [8000, 86],
        [9000, 86],
        [10000, 85],
        [12000, 95],
        [15000, 110],
        [20000, 125],
        [24000, 140],
    ]
)

INVERSE_EQUAL_LOUDNESS_CURVE = AVERAGE_EQUAL_LOUDNESS_CURVE.copy()
INVERSE_EQUAL_LOUDNESS_CURVE[:, 1] = 70 - INVERSE_EQUAL_LOUDNESS_CURVE[:, 1]

MIN_PITCH = 21
MAX_PITCH = 108


MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11])
MINOR_SCALE = np.array([0, 2, 3, 5, 7, 8, 10])
