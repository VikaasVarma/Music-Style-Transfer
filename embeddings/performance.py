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

import uuid
import numpy as np
import pretty_midi as pm

from midi import Midi


class Performance:
    # Input to the performance encoder as a one hot encoding of 415 events
    # 128 Note On, 128 Note Off, 125 Time-Shift (0-1s), 32 Velocity, 2 Pedal
    # https://arxiv.org/pdf/1808.03715.pdf
    encoding_type = "performance"

    def __init__(self, midi: Midi):
        self.notes = midi.piano.notes
        self.control_changes = midi.piano.control_changes

        self.load()

    def load(self):
        # First round all times to the nearest 8ms
        for note in self.notes:
            note.discrete_start = round(note.start * 125)
            note.discrete_end = round(note.end * 125)

        # Separate note on and note off
        notes = (
            [
                {
                    "event": "note_on",
                    "time": note.discrete_start,
                    "exact_time": note.start,
                    "pitch": note.pitch,
                    "velocity": note.velocity // 4,
                }
                for note in self.notes
            ]
            + [
                {
                    "event": "note_off",
                    "time": note.discrete_end,
                    "exact_time": note.end,
                    "pitch": note.pitch,
                }
                for note in self.notes
            ]
            + [
                {
                    "event": "sustain",
                    "time": round(cc.time * 125),
                    "exact_time": cc.time,
                    "value": cc.value,
                }
                for cc in self.control_changes
                if cc.number >= 64
            ]
        )
        notes.sort(key=lambda x: x["time"])

        # Convert into a list of events
        # Note exact time of time shifts is the time after the shift
        events = [("time_shift", notes[0]["time"])]
        exact_times = [notes[0]["exact_time"]]
        velocity = 0
        pedal = False
        for e1, e2 in zip(notes, notes[1:]):
            if e1["event"] == "note_on":
                if e1["velocity"] != velocity:
                    velocity = e1["velocity"]
                    events.append(("set_velocity", velocity))
                    exact_times.append(e1["exact_time"])
                events.append(("note_on", e1["pitch"]))
                exact_times.append(e1["exact_time"])
            if e1["event"] == "note_off":
                events.append(("note_off", e1["pitch"]))
                exact_times.append(e1["exact_time"])
            if e1["event"] == "sustain":
                if (e1["value"] >= 64) != pedal:
                    pedal = e1["value"] >= 64
                    events.append(("sustain", 1 if pedal else 0))
                    exact_times.append(e1["exact_time"])

            while e2["time"] - e1["time"] > 125:
                events.append(("time_shift", 124))
                exact_times.append(e1["exact_time"] + 1)
                e1["time"] += 125
                e1["exact_time"] += 1

            if e2["time"] == e1["time"]:
                continue

            events.append(("time_shift", e2["time"] - e1["time"] - 1))
            exact_times.append(e2["exact_time"])

        # Convert into a one hot encoding
        self.encoding = np.zeros((len(events), 415))
        self.exact_times = np.array(exact_times)
        offsets = {
            "note_on": 0,
            "note_off": 128,
            "time_shift": 256,
            "set_velocity": 381,
            "sustain": 413,
        }
        for encoding, (event, value) in zip(self.encoding, events):
            encoding[offsets[event] + value] = 1

    def save(encoding: np.ndarray, filename: str = "temp.mid"):
        midi = Performance.performance_to_midi(encoding)
        midi.save(filename)

    @staticmethod
    def performance_to_midi(encoding: np.ndarray) -> Midi:
        def decode(event):
            offsets = [
                (413, "sustain"),
                (381, "set_velocity"),
                (256, "time_shift"),
                (128, "note_off"),
                (0, "note_on"),
            ]

            for offset, name in offsets:
                if event >= offset:
                    return name, event - offset

        notes = []
        control_changes = []
        start = [0 for _ in range(128)]
        time = 0
        velocity = 0
        for event in encoding.T:
            event, value = decode(np.argmax(event))
            if event == "note_on":
                start[value] = time
            if event == "note_off":
                notes.append(pm.Note(velocity * 4, value, start[value], time))
            if event == "time_shift":
                time += value / 125
            if event == "set_velocity":
                velocity = value
            if event == "sustain":
                control_changes.append(pm.ControlChange(64, 63 + value, time))

        instrument = pm.Instrument(0)
        instrument.notes = notes
        instrument.control_changes = control_changes
        midi = pm.PrettyMIDI()
        midi.instruments.append(instrument)
        midi.time_signature_changes = [pm.TimeSignature(4, 4, 0)]
        return Midi(midi, None, None, None, uuid.uuid4().int & ((1 << 32) - 1))


if __name__ == "__main__":
    from midi import Midi

    midi = Midi.get_sample_from_maestro(1)
    performance = Performance(midi)
