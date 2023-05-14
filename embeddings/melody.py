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

from __future__ import annotations
import numpy as np
from scipy import signal
import pretty_midi as pm

from midi import Midi

from constants import (
    INVERSE_EQUAL_LOUDNESS_CURVE,
    MIN_PITCH,
    MAX_PITCH,
    MAJOR_SCALE,
    MINOR_SCALE,
)

from hyperparameters import (
    SAMPLES_PER_BEAT,
    THRESHOLD_FACTOR,
    THRESHOLD_DEVIATION,
    VOICING_THRESHOLD_DEVIATION,
)


class Melody:
    encoding_type = "melody"

    def __init__(self, midi: Midi):
        self.midi = midi
        self.fs = self.midi.fs

        # Initialize melody for extraction
        self.melody = self.midi.piano_roll
        self.extract_melody()
        self.encode_melody()

    def save(self, filename: str = "temp_melody.mid") -> None:
        instrument = pm.Instrument(0)

        # Run length encode piano roll
        diff = np.c_[
            self.melody[:, 1:] != self.melody[:, :-1], np.ones(128, dtype=bool)
        ]
        for pitch, row in enumerate(diff):
            velocities = self.melody[pitch, row].astype(int)
            starts = np.append(0, np.where(row)[0] + 1)
            notes = [
                pm.Note(velocity, pitch, start / self.fs, end / self.fs)
                for velocity, start, end in zip(velocities, starts, starts[1:])
            ]
            instrument.notes.extend(notes)

        midi = pm.PrettyMIDI()
        midi.instruments.append(instrument)
        midi.write(filename)

    # Applies an equal loudness filter to the piano roll
    # https://replaygain.hydrogenaud.io/equal_loudness.html
    def apply_equal_loudness_filter(self):
        # Change units inverse equal loudness curve from frequency to midi pitch
        ielc = INVERSE_EQUAL_LOUDNESS_CURVE.copy()
        ielc[:, 0] = 12 * np.log2(ielc[:, 0] / 440) + 69

        # Interpolate the inverse equal loudness curve to the piano roll
        loudness_vector = np.interp(np.arange(128), *ielc.T)

        # Change units from ∆dB to ∆(Midi Velocity)
        loudness_vector = np.power(10, loudness_vector / 40)

        # Apply filter for each time step
        self.melody = np.maximum(self.melody * loudness_vector[:, None], 0)

    def denoise(
        self,
        THRESHOLD_FACTOR: float = THRESHOLD_FACTOR,
        THRESHOLD_DEVIATION: float = THRESHOLD_DEVIATION,
    ) -> None:
        # Operate only on time steps with notes
        mask = np.any(self.melody > 0, axis=0)

        THRESHOLD_FACTORS = np.max(self.melody, axis=0) * THRESHOLD_FACTOR
        self.melody[self.melody < THRESHOLD_FACTORS] = 0

        # Update mask
        mask = np.any(self.melody > 0, axis=0)

        std = np.std(self.melody[:, mask], axis=0, where=self.melody[:, mask] > 0)
        mean = np.mean(self.melody[:, mask], axis=0, where=self.melody[:, mask] > 0)

        THRESHOLD_DEVIATIONS = mean - std * THRESHOLD_DEVIATION
        self.melody[:, mask] = np.where(
            self.melody[:, mask] < THRESHOLD_DEVIATIONS, 0, self.melody[:, mask]
        )

    def detect_voices(self):
        # Notes with higher salience are more likely to be in the melody
        average_salience = np.mean(self.melody, where=self.melody > 0)
        std_salience = np.std(self.melody, where=self.melody > 0)
        threshold = average_salience - std_salience * VOICING_THRESHOLD_DEVIATION
        self.melody[self.melody < threshold] = 0

    def remove_outliers(self):
        mask = np.any(self.melody > 0, axis=0)
        mean = np.arange(128) @ self.melody[:, mask] / self.melody[:, mask].sum(axis=0)

        # Calculate moving average of 5 seconds
        window = int(5 * self.fs)
        ma = np.zeros(self.melody.shape[1])
        ma[mask] = signal.convolve2d(
            mean[:, None], np.ones((window, 1)) / window, mode="same", boundary="symm"
        ).reshape(-1)

        # Remove octave duplicates
        for avg, frame in zip(ma, self.melody.T):
            pitches = frame.nonzero()[0]
            for pitch in pitches[np.where(np.diff(pitches) == 12)[0]]:
                # Removing the note that is farther from the moving average
                frame[max(pitch, pitch + 12, key=lambda p: abs(p - avg))] = 0

        # Recompute mean and ma
        mean = np.arange(128) @ self.melody[:, mask] / self.melody[:, mask].sum(axis=0)
        ma = np.convolve(mean, np.ones(window) / window, mode="same")

        # Remove pitches more than an octave away from the moving average
        for avg, frame in zip(ma, self.melody.T):
            pitches = frame.nonzero()[0][frame.nonzero()[0] - avg > 12]
            frame[pitches] = 0

    def select_melody(self):
        self.melody[self.melody < np.max(self.melody, axis=0)] = 0
        self.melody = np.argmax(self.melody, axis=0)

        # Ensure melody lies within expected pitch range
        assert np.all(self.melody[self.melody != 0] >= MIN_PITCH) and np.all(
            self.melody[self.melody != 0] < MAX_PITCH
        ), f"Melody from {self.midi.filepath} contains pitch out of range: \
            {np.min(self.melody[self.melody != 0])}, {np.max(self.melody)}"

    # Extracts the melody of a midi file
    # Following the information in this paper modified to accomadate midi file input (rather than wav)
    # http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamongomezmelodytaslp2012.pdf
    def extract_melody(self):
        # Apply equal loudness filter to piano roll
        self.apply_equal_loudness_filter()

        # Denoise to make pitch contours more prominent
        self.denoise()

        # Detect and prune voices
        self.detect_voices()

        # Remove octave errors and pitch outliers
        self.remove_outliers()

        # Select melody
        self.select_melody()

    def get_key_counts(self, melody: np.ndarray, key: np.ndarray) -> np.ndarray:
        note_counts = np.bincount(melody % 12, minlength=12)
        key_counts = np.zeros(12)
        for note, count in enumerate(note_counts):
            key_counts[(key + note) % 12] += count
        return key_counts

    def encode_melody(self):
        # Create vector embedding for each time step taking into account
        # pitch, note playing, note attack, ascending/descending, repeating, position in bar, and some measure of key
        offset = MAX_PITCH - MIN_PITCH
        self.encoding = np.zeros((len(self.melody), offset + 39))

        for i, pitch in enumerate(self.melody):
            embedding = np.zeros(offset + 39)
            if pitch != 0:
                embedding[pitch - MIN_PITCH] = 1
                embedding[offset] = 1

            if (i == 0 and pitch != 0) or pitch != self.melody[i - 1]:
                embedding[offset + 1] = 1
            else:
                embedding[offset + 2] = 1

            if i > 0:
                embedding[int(pitch > self.melody[i - 1]) + offset + 3] = 1

            for j in range(2):
                # Looking back j measures to see if the note is the same
                pos = i - (j + 1) * SAMPLES_PER_BEAT * self.midi.time_signature[0]
                if pos >= 0 and pitch == self.melody[pos]:
                    embedding[j + offset + 5] = 1

            # Storing position in bar
            loc = format(i % (SAMPLES_PER_BEAT * self.midi.time_signature[0]), "05b")
            loc = [int(j) for j in loc[-5:]]
            embedding[offset + 7 : offset + 12] = loc  # noqa: E203

            if i % (SAMPLES_PER_BEAT * self.midi.time_signature[0]) == 0:
                embedding[offset + 14] = 1

            # Provide a measure of the extent to which the melody falls into each key
            for j, scale in zip((offset + 15, offset + 27), (MAJOR_SCALE, MINOR_SCALE)):
                counts = self.get_key_counts(self.melody[:j], scale)
                embedding[np.argwhere(counts == np.max(counts)) + j] = 1

            self.encoding[i] = embedding

    def _get_levenstein_dist(self, other: Melody) -> float:
        # Calculates the levenstein distance between two melodies
        edit_dist = np.zeros((len(self.melody) + 1, len(other.melody) + 1))
        edit_dist[:, 0] = np.arange(len(self.melody) + 1)
        edit_dist[0] = np.arange(len(other.melody) + 1)

        for i, p1 in enumerate(self.melody):
            for j, p2 in enumerate(other.melody):
                edit_dist[i + 1, j + 1] = min(
                    edit_dist[i, j + 1] + 12,
                    edit_dist[i + 1, j] + 12,
                    edit_dist[i, j] + min(abs(p1 - p2) % 12, 12 - abs(p1 - p2) % 12),
                )

        return edit_dist[-1, -1]

    def get_dist(self, other: Melody, dist_type: str = "levenstein") -> float:
        if dist_type == "levenstein":
            return self._get_levenstein_dist(other)
