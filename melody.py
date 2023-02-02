import numpy as np
from scipy import signal
import pretty_midi as pm

from midi import Midi

from constants import (
    inverse_equal_loudness_curve,
    threshold_factor,
    threshold_deviation,
    voicing_threshold_deviation,
)


class Melody:
    def __init__(self, midi: Midi):
        self.midi = midi

        self.load_piano_roll()
        self.extract_melody()

    def load_piano_roll(self, spb: float = 16):
        # spb = samples per beat
        self.fs = spb * self.midi.tempo / 60
        self.melody = self.midi.get_piano_roll(self.fs)

    def save(self, filename: str = "temp_melody.mid"):
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
        ielc = inverse_equal_loudness_curve.copy()
        ielc[:, 0] = 12 * np.log2(ielc[:, 0] / 440) + 69

        # Interpolate the inverse equal loudness curve to the piano roll
        loudness_vector = np.interp(np.arange(128), *ielc.T)

        # Change units from ∆dB to ∆(Midi Velocity)
        loudness_vector = np.power(10, loudness_vector / 40)

        # Apply filter for each time step
        self.melody = np.maximum(self.melody * loudness_vector[:, None], 0)

    def denoise(
        self,
        threshold_factor: float = threshold_factor,
        threshold_deviation: float = threshold_deviation,
    ) -> np.ndarray:

        # Operate only on time steps with notes
        mask = np.any(self.melody > 0, axis=0)

        threshold_factors = np.max(self.melody, axis=0) * threshold_factor
        self.melody[self.melody < threshold_factors] = 0

        # Update mask
        mask = np.any(self.melody > 0, axis=0)

        std = np.std(self.melody[:, mask], axis=0, where=self.melody[:, mask] > 0)
        mean = np.mean(self.melody[:, mask], axis=0, where=self.melody[:, mask] > 0)

        threshold_deviations = mean - std * threshold_deviation
        self.melody[:, mask] = np.where(
            self.melody[:, mask] < threshold_deviations, 0, self.melody[:, mask]
        )

    def detect_voices(self):
        # Notes with higher salience are more likely to be in the melody
        average_salience = np.mean(self.melody, where=self.melody > 0)
        std_salience = np.std(self.melody, where=self.melody > 0)
        threshold = average_salience - std_salience * voicing_threshold_deviation
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
        test = self.melody.copy()
        for avg, frame in zip(ma, self.melody.T):
            pitches = frame.nonzero()[0][frame.nonzero()[0] - avg > 12]
            frame[pitches] = 0

    def select_melody(self):
        self.melody[self.melody < np.max(self.melody, axis=0)] = 0

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


melody = Melody(Midi.get_sample(100))
melody.save()
self = melody
# print(melody.midi.composer, melody.midi.title)
