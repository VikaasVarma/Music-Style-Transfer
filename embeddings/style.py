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

import contextlib
import json
import os
import numpy as np
from numba.core.errors import NumbaDeprecationWarning
import warnings

from midi import Midi
from constants import VAMP_PATH, CHORD_ENCODINGS_PATH

os.environ["VAMP_PATH"] = VAMP_PATH
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

from chord_extractor.extractors import Chordino  # noqa


class Style:
    encoding_type = "style"

    def __init__(self, midi: Midi):
        self.midi = midi

        # File that stores encodings of chords based on vocabulary
        if not os.path.exists(CHORD_ENCODINGS_PATH):
            with open(CHORD_ENCODINGS_PATH, "w") as f:
                json.dump([1, {"N": 0}], f)

        with open(CHORD_ENCODINGS_PATH, "r") as f:
            self.num_chords, self.vocab = json.load(f)

        # Silence output from Chordino
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                self.extract_style()
                self.generate_encoding()

    def extract_style(self):
        chordino = Chordino()
        conversion_file_path = chordino.preprocess(self.midi.filepath)

        chords = chordino.extract(conversion_file_path)
        self.chords = [(c.chord, c.timestamp) for c in chords]

    def generate_encoding(self):
        # One hot encodes all chords in the midi
        self.encoding = np.zeros((self.midi.piano_roll.shape[1], self.num_chords))

        for (chord, t1), (chord, t2) in zip(self.chords, self.chords[1:]):
            if chord not in self.vocab:
                self.vocab[chord] = self.num_chords
                self.num_chords += 1

                self.encoding = np.hstack(
                    (self.encoding, np.zeros((self.encoding.shape[0], 1)))
                )

            start = self.midi.time_to_tick(t1)
            end = self.midi.time_to_tick(t2)
            index = self.vocab[chord]

            self.encoding[start:end, index] = 1

        with open(CHORD_ENCODINGS_PATH, "w") as f:
            json.dump([self.num_chords, self.vocab], f)
