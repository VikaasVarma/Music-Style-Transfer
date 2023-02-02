import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi as pm


# Useful util functions for working with MIDI files
maestro = pd.read_csv("maestro-v3.0.0/maestro-v3.0.0.csv")


class Midi:
    def __init__(self, midi: pm.PrettyMIDI, filepath: str, composer: str, title: str):
        self.midi = midi
        self.filepath = filepath
        self.composer = composer
        self.title = title

        # Ensure sample is monophonic piano and load useful attributes
        assert len(midi.instruments) == 1
        assert midi.instruments[0].program == 0
        self.piano: pm.Instrument = midi.instruments[0]

    def get_piano_roll(self, fs: float) -> np.ndarray:
        return self.piano.get_piano_roll(fs)

    def plot(self, spb: float = 16):
        piano_roll = self.get_piano_roll(spb * self.tempo / 60)
        plt.imshow(piano_roll[:, ::spb])
        plt.show()

    @property
    def tempo(self) -> float:
        return self.midi.estimate_tempo()

    @classmethod
    def get_sample(cls, id: int = -1) -> pm.PrettyMIDI:
        if id < 0:
            id = maestro.sample().index[0]
        filepath = os.path.join("maestro-v3.0.0", maestro.at[id, "midi_filename"])

        midi = pm.PrettyMIDI(filepath)

        return cls(
            midi,
            filepath,
            maestro.at[id, "canonical_composer"],
            maestro.at[id, "canonical_title"],
        )
