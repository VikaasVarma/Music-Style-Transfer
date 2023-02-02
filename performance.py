from typing import List
import numpy as np
import pretty_midi as pm


class Performance:
    # Input to the performance encoder as a one hot encoding of 415 different events
    # 128 Note On, 128 Note Off, 125 Time-Shift (dividing 1s), 32 Velocity, 2 Pedal
    # https://arxiv.org/pdf/1808.03715.pdf
    def __init__(self, notes: List[pm.Note], control_changes: List[pm.ControlChange]):
        self.notes = notes
        self.control_changes = control_changes

        self.load()

    def load(self):
        # First round all times to the nearest 8ms
        for note in self.notes:
            note.start = round(note.start * 125) / 125
            note.end = round(note.end * 125) / 125

        # Separate note on and note off
        notes = (
            [
                {
                    "event": "note_on",
                    "time": note.start,
                    "pitch": note.pitch,
                    "velocity": note.velocity // 4,
                }
                for note in self.notes
            ]
            + [
                {
                    "event": "note_off",
                    "time": note.end,
                    "pitch": note.pitch,
                    "velocity": note.velocity // 4,
                }
                for note in self.notes
            ]
            + [
                {"event": "sustain", "time": cc.time, "value": cc.value}
                for cc in self.control_changes
                if cc.number == 64
            ]
        )
        notes.sort(key=lambda x: x["time"])

        # Convert into a list of events
        events = [("time_shift", int(notes[0]["time"] * 125))]
        velocity = 0
        pedal = False
        for e1, e2 in zip(notes, notes[1:]):
            match e1["event"]:
                case "note_on":
                    if e1["velocity"] != velocity:
                        velocity = e1["velocity"]
                        events.append(("set_velocity", velocity))
                    events.append(("note_on", e1["pitch"]))
                case "note_off":
                    events.append(("note_off", e1["pitch"]))
                case "sustain":
                    if e1["value"] >= 64 != pedal:
                        pedal = e1["value"] >= 64
                        events.append(("sustain", 1 if pedal else 0))

            while e2["time"] - e1["time"] > 1:
                events.append(("time_shift", 124))
                e1["time"] += 1

            events.append(("time_shift", int((e2["time"] - e1["time"]) * 125) - 1))

        # Convert into a one hot encoding
        self.encoding = np.zeros((415, len(events)))
        offsets = {
            "note_on": 0,
            "note_off": 128,
            "time_shift": 256,
            "set_velocity": 381,
            "sustain": 413,
        }
        for i, (event, value) in enumerate(events):
            self.encoding[offsets[event] + value, i] = 1

    def save(encoding: np.ndarray, filename: str = "temp.mid"):
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
            match decode(np.argmax(event)):
                case ("note_on", pitch):
                    start[pitch] = time
                case ("note_off", pitch):
                    notes.append(pm.Note(velocity * 4, pitch, start[pitch], time))
                case ("time_shift", time):
                    time += time / 125
                case ("set_velocity", vel):
                    velocity = vel
                case ("sustain", value):
                    control_changes(pm.ControlChange(64, 63 + value, time))

        instrument = pm.Instrument(0)
        instrument.notes = notes
        instrument.control_changes = control_changes
        midi = pm.PrettyMIDI()
        midi.instruments.append(instrument)
        midi.write(filename)
