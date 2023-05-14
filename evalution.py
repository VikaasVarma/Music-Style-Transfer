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

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from statistics import NormalDist

from embeddings.performance import Performance
from embeddings.melody import Melody


def get_velocity_duration_evaluation(compressed: torch.Tensor):
    velocities = []
    note_durations = []
    active_notes = {}
    sustained_notes = set()

    sustain = False
    velocity = 0
    time = 0
    for event in compressed:
        event = event.item()
        if event < 128:
            velocities.append(velocity)
            if event not in active_notes:
                active_notes[event] = time
        if 128 <= event < 256:
            if event - 128 in active_notes:
                # Only add to active notes if note was previously pressed and sustained
                if sustain and event - 128 in active_notes:
                    active_notes[event - 128] = time
                else:
                    note_durations.append(time - active_notes[event - 128])
                    del active_notes[event - 128]
        if 256 <= event < 381:
            time += (event - 255) / 125
        if 381 <= event < 413:
            velocity = event - 381
        if 413 <= event:
            sustain = event == 414
            if not sustain:
                for note in sustained_notes:
                    note_durations.append(time - active_notes[note])
                    del active_notes[note]

    metrics = {
        "mean_velocity": np.mean(velocities),
        "var_velocity": np.var(velocities),
        "mean_duration": np.mean(note_durations),
        "var_duration": np.var(note_durations),
    }
    return metrics, time


def evaluate_batch(performance: torch.Tensor):
    # performance has shape B, S, E

    # Calculate following metrics:
    # - Note density
    # - Pitch range
    # - Mean pitch, Variance of pitch
    # - Mean velocity, Variance of velocity
    # - Mean duration, Variance of duration

    metrics = {}
    compressed = performance.argmax(dim=-1)  # B, S

    # number of note on events
    num_notes = performance[:, :, :128].sum(dim=(1, 2))

    # pitches of note on events
    notes = [b[b < 128].float() for b in compressed]

    vel_duration_evals, times = zip(
        *[get_velocity_duration_evaluation(b) for b in compressed]
    )

    metrics["note_density"] = num_notes.cpu().numpy() / np.array(times)
    metrics["pitch_range"] = np.array([(b.max() - b.min()).item() for b in notes])
    metrics["mean_pitch"] = np.array([b.mean().item() for b in notes])
    metrics["var_pitch"] = np.array([b.var().item() for b in notes])

    metrics.update(
        {
            metric: [evals[metric] for evals in vel_duration_evals]
            for metric in vel_duration_evals[0]
        }
    )

    return metrics


def overlapping_area(pred: pd.DataFrame, target: pd.DataFrame):
    # Fits pred and target to gaussian and computes overlapping area
    means = [pred.mean(), target.mean()]
    stds = [pred.std(), target.std()]
    return {
        metric: NormalDist(m1, s1).overlap(NormalDist(m2, s2))
        for m1, m2, s1, s2, metric in zip(*means, *stds, pred.columns)
    }


def evaluate_melody_similarity(pred: torch.Tensor, target: torch.Tensor):
    # Evaluate on batches of shape B, S, E
    preds = pred.argmax(dim=-1)
    targets = target.argmax(dim=-1)

    preds = nn.functional.one_hot(preds, num_classes=417)
    targets = nn.functional.one_hot(targets, num_classes=417)

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    dists = np.zeros(len(preds))
    for i, (pred, target) in enumerate(zip(preds, targets)):
        pred = Performance.performance_to_midi(pred)
        target = Performance.performance_to_midi(target)

        pred = Melody(pred)
        target = Melody(target)

        dists[i] = pred.get_dist(target)

    return dists


if __name__ == "__main__":
    from preprocess import load_json, expand_encoding

    a = load_json(1, "performance")
    b = load_json(0, "performance")
    a = expand_encoding(a)
    b = expand_encoding(b)
    A = torch.from_numpy(a).unsqueeze(0)
    B = torch.from_numpy(a).unsqueeze(0)

    sim = evaluate_melody_similarity(A, B)
    print(sim)
