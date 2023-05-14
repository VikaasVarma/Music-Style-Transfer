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

import os

import pandas as pd
import pretty_midi as pm

from constants import MAESTRO_DATA_PATH
from hyperparameters import SAMPLES_PER_BEAT


class Midi:
    def __init__(
        self, midi: pm.PrettyMIDI, filepath: str, composer: str, title: str, idx: int
    ):
        self.midi = midi
        self.filepath = filepath
        self.composer = composer
        self.title = title
        self.idx = idx
        self.duration = midi.get_end_time()

        # Ensure midi uses only one time signature
        assert (
            len(self.midi.time_signature_changes) == 1
        ), f"Multiple time changes in midi {filepath}"
        self.time_signature = (
            self.midi.time_signature_changes[0].numerator,
            self.midi.time_signature_changes[0].denominator,
        )

        # Samples the MIDI
        self.tempo = midi.estimate_tempo()
        self.fs = self.tempo / 60 * SAMPLES_PER_BEAT

        # Ensure sample is monophonic piano and load useful attributes
        assert len(midi.instruments) == 1, "Multiple instruments in midi"
        assert midi.instruments[0].program == 0, "Instrument is not piano"

        self.piano: pm.Instrument = midi.instruments[0]
        self.piano_roll = self.piano.get_piano_roll(self.fs)

    def time_to_tick(self, time):
        return int(time * self.fs)

    def save(self, filepath: str):
        self.filepath = filepath
        self.midi.write(filepath)

    @classmethod
    def get_sample_from_maestro(cls, idx: int = -1):
        maestro = pd.read_csv(MAESTRO_DATA_PATH)
        if idx < 0:
            idx = maestro.sample().index[0]
        filepath = os.path.join(
            "data", "maestro-v3.0.0", maestro.at[idx, "midi_filename"]
        )

        midi = pm.PrettyMIDI(filepath)

        return cls(
            midi,
            filepath,
            maestro.at[idx, "canonical_composer"],
            maestro.at[idx, "canonical_title"],
            idx,
        )
