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

import math
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_optimizer import Adafactor
import wandb

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from evalution import evaluate_batch, overlapping_area, evaluate_melody_similarity
from constants import DEVICE, MODEL_PATH
from hyperparameters import (
    BATCH_SIZE,
    DEFAULT_CONFIG,
    LEARNING_RATE,
    MINI_BATCH_SIZE,
    NUM_WARMUP_STEPS,
)
from model.dataloader import train_test_val_split
from model.transformer_autoencoder import MusicTransformerAutoencoder


def lr_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    steps_per_epoch: int,
    epochs_per_cycle: int,
):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps

        step -= num_warmup_steps
        cos = math.cos(math.pi * step / steps_per_epoch / epochs_per_cycle)
        cycle_parity = 1 if step // (steps_per_epoch * epochs_per_cycle) % 2 else -1
        return (1.001 - cos * cycle_parity) / 2

    return LambdaLR(optimizer, lr_lambda)


def mini_batch(model, inputs, target, loss_fn):
    pred = model(*inputs)
    _, targets = torch.max(target, dim=-1)
    loss = loss_fn(pred.flatten(0, 1), targets.flatten())
    return loss


def run_epoch(
    model: MusicTransformerAutoencoder, train_loader, loss_fn, optimizer, scheduler
):
    model.train()
    optimizer.zero_grad()
    loss_fn.to(DEVICE)
    total_loss = 0
    batch_loss = 0
    with tqdm(
        total=len(train_loader) // (BATCH_SIZE // MINI_BATCH_SIZE),
        desc="Batch",
        position=1,
        leave=False,
    ) as pbar:
        for i, (inputs, target) in enumerate(train_loader):
            loss = mini_batch(model, inputs, target, loss_fn)
            total_loss += loss
            batch_loss += loss
            loss.backward()

            if (i + 1) % (BATCH_SIZE // MINI_BATCH_SIZE) == 0:
                wandb.log({"batch_loss": batch_loss.item()})
                batch_loss = 0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)

    return total_loss.item() / len(train_loader)


def run_validation(model, val_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        loss = 0
        for inputs, target in tqdm(val_loader, desc="Batch", position=1, leave=False):
            loss += mini_batch(model, inputs, target, loss_fn).item()

    return loss / len(val_loader)


def evaluate(model, test_loader):
    model.eval()
    columns = [
        "note_density",
        "pitch_range",
        "mean_pitch",
        "var_pitch",
        "mean_velocity",
        "var_velocity," "mean_duration",
        "var_duration",
    ]

    pred_metrics = pd.DataFrame(columns=columns)
    target_metrics = pd.DataFrame(columns=columns)
    melody_distances = []

    for (melody, _, perf1), _ in tqdm(
        test_loader, desc="Batch", position=1, leave=False
    ):
        (_, style, perf2), _ = test_loader.dataset[np.random.choice(len(test_loader))]
        with torch.no_grad():
            pred = model.generate(melody, style, perf1, perf2)
        pred_metrics = pd.concat(
            [pred_metrics, pd.DataFrame(evaluate_batch(pred))],
            ignore_index=True,
        )
        target_metrics = pd.concat(
            [target_metrics, pd.DataFrame(evaluate_batch(style))],
            ignore_index=True,
        )

        melody_distances.append(evaluate_melody_similarity(pred, melody))

    metrics = overlapping_area(pred_metrics, target_metrics)
    metrics["melody_similarity"] = np.mean(melody_distances)
    return metrics


def save_model(model, epoch, title, best=False):
    if best:
        title = "best"
        epoch = 0
    torch.save(
        dict({"epoch": epoch, "state": model.state_dict()}, **model.config()),
        os.path.join(MODEL_PATH, f"{title}-{epoch}.pt"),
    )


def remove_old_models(epoch):
    if epoch - 2 >= 0 and epoch % 5 != 0:
        os.remove(os.path.join(MODEL_PATH, f"music-transformer-{epoch-2}.pt"))


def train(model, num_epochs):
    train_loader, val_loader, test_loader = train_test_val_split()
    optimizer = Adafactor(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_schedule(
        optimizer, NUM_WARMUP_STEPS, len(train_loader) // BATCH_SIZE, 10
    )
    loss_fn = nn.CrossEntropyLoss()
    best_loss = np.inf

    for epoch in tqdm(range(num_epochs), desc="Epoch", position=0, leave=False):
        train_loader.dataset.shuffle()

        training_loss = run_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        wandb.log({"train_loss": training_loss})

        torch.cuda.empty_cache()

        validation_loss = run_validation(model, val_loader, loss_fn)
        wandb.log({"validation_loss": validation_loss})

        torch.cuda.empty_cache()

        # metrics = evaluate(model, test_loader)
        # wandb.log(metrics)

        save_model(model, epoch, "music-transformer")
        remove_old_models(epoch)

        if validation_loss < best_loss:
            best_loss = validation_loss
            save_model(model, epoch, "", best=True)


def tune_hyperparameters(num_epochs=5, **hyperparameters):
    # Takes in a dictionary of hyperparameters and set of values for each
    # Trains the model for num_epochs for each combination of hyperparameters
    with tqdm(
        total=sum(map(len, hyperparameters.values())),
        desc="Hyperparameter",
        position=-1,
    ) as pbar:
        for param, values in hyperparameters.items():
            for value in values:
                pbar.set_description(f"Tuning {param} at {value}")
                # Initialize model with default parameters, changing only one at a time
                model = MusicTransformerAutoencoder(**{param: value})
                model.to(DEVICE)

                # Initialize wandb run for easy visualization and logging
                wandb.init(
                    project="music-transformer-autoencoder",
                    config=dict(DEFAULT_CONFIG, param=value),
                )

                train(model, num_epochs)
                pbar.update(1)


if __name__ == "__main__":
    wandb.init(
        project="music-transformer-autoencoder",
        config=DEFAULT_CONFIG,
    )

    # Initialize with all default hyperparameters
    model = MusicTransformerAutoencoder()
    model.to(DEVICE)
    train(model, 50)
