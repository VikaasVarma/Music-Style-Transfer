import matplotlib.pyplot as plt

from midi import get_piano_roll


def plot_piano_roll(sample, spb=16):
    piano_roll = get_piano_roll(sample, spb)
    plt.imshow(piano_roll[:, ::spb])
    plt.show()
