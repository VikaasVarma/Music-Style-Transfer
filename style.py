from midi import Midi
from pychord import Chord
from chord_extractor.extractors import Chordino




def extract_style(midi: Midi):
    chordino = Chordino(roll_on=1)
    conversion_file_path = chordino.preprocess(midi.filepath)

    chords = chordino.extract(conversion_file_path)["list"]
    processed = [(Chord(chords['label']), chords['timestamp']) for chord in chords if chord['label'] != 'N']

"""
Chord properties to one hot encode:
Root nodes
Chord type (major, minor, augmented, suspended, diminished, number of notes, inversions)
    - major


"""