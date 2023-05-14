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

import matplotlib.pyplot as plt
import numpy as np
from midi import Midi
from embeddings.melody import Melody


def plot_piano_roll(sample: Midi):
    plt.imshow(sample.piano_roll)
    plt.ylim(0, 128)
    plt.xlim(0, min(sample.piano_roll.shape[1] // 10, 250))
    plt.show()


def plot_melody_steps(sample: Midi):
    _, ax = plt.subplots(2, 3)
    ax = ax.flatten()
    piano_roll = sample.piano_roll

    melody = Melody(sample)
    melody.melody = piano_roll
    ax[0].imshow(melody.melody)
    ax[0].set_ylim(0, 128)
    ax[0].set_xlim(0, min(melody.melody.shape[1] // 10, 250))

    melody.apply_equal_loudness_filter()
    ax[1].imshow(melody.melody)
    ax[1].set_ylim(0, 128)
    ax[1].set_xlim(0, min(melody.melody.shape[1] // 10, 250))

    melody.denoise()
    ax[2].imshow(melody.melody)
    ax[2].set_ylim(0, 128)
    ax[2].set_xlim(0, min(melody.melody.shape[1] // 10, 250))

    melody.detect_voices()
    ax[3].imshow(melody.melody)
    ax[3].set_ylim(0, 128)
    ax[3].set_xlim(0, min(melody.melody.shape[1] // 10, 250))

    melody.remove_outliers()
    ax[4].imshow(melody.melody)
    ax[4].set_ylim(0, 128)
    ax[4].set_xlim(0, min(melody.melody.shape[1] // 10, 250))

    melody.select_melody()
    one_hot = np.eye(128)[melody.melody].T
    ax[5].imshow(one_hot)
    ax[5].set_ylim(0, 128)
    ax[5].set_xlim(0, min(one_hot.shape[1] // 10, 250))

    plt.show()


def plot_loss(losses, labels):
    x = np.linspace(0, 10, losses.shape[0])
    plt.plot(x, losses, label=labels)
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, 15)
    plt.show()


def plot_loss_val(losses, labels):
    x = np.linspace(0, 10, losses.shape[0])
    plt.plot(x, losses[:, :3], label=labels)
    plt.legend()
    plt.gca().set_prop_cycle(None)
    plt.plot(x, losses[:, 3:], ":")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, 12)
    plt.show()
