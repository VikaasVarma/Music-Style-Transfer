import pretty_midi as pm

from midi import Midi
from performance import Performance
from melody import Melody
from style import extract_style


class Encoding:
    def __init__(self, midi: Midi):
        self.midi = midi

    @property
    def performance(self):
        if not hasattr(self, "_performance"):
            self._performance = Performance(
                self.midi.piano.notes, self.midi.piano.control_changes
            )
        return self._performance

    @property
    def melody(self):
        if not hasattr(self, "_melody"):
            self._melody = Melody(self.midi.midi)
        return self._melody

    @property
    def style(self):
        if not hasattr(self, "_style"):
            self._style = extract_style(self.midi)
        return self._style
