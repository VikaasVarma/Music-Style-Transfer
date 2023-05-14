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

from functools import partial
import os
import torch
import torch.nn as nn
from tqdm import tqdm


from constants import DEVICE, PROCESSED_DATA_PATH
from hyperparameters import (
    ATTENTION_HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_HEADS,
    DIM_FF,
    MAX_SEQ_LEN,
    DROPOUT_PROB,
)

from preprocess import expand_encoding, load_metadata, load_json, process_midi
from midi import Midi
from embeddings.melody import Melody
from embeddings.style import Style
from embeddings.performance import Performance

from model.positional_encoder import PositionalEncoder
from model.transformer_components import TransformerEncoder, TransformerDecoder


class MusicTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        device=DEVICE,
        d_model=ATTENTION_HIDDEN_SIZE,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        d_hidden=DIM_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT_PROB,
    ):
        super().__init__()

        self.metadata = load_metadata(PROCESSED_DATA_PATH)
        self.device = device
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.max_seq_len = max_seq_len

        self.token_encoders = nn.ModuleList(
            [
                nn.Linear(d_embd, d_model)
                for d_embd in (
                    self.metadata["d_embd_melody"],
                    self.metadata["d_embd_style"],
                    self.metadata["d_embd_performance"],
                    self.metadata["d_embd_performance"],
                )
            ]
        )

        self.pos_encoders = nn.ModuleList(
            [PositionalEncoder(d_model, max_seq_len, dropout) for _ in range(4)]
        )

        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    d_model, n_layers, n_heads, d_hidden, max_seq_len, dropout
                )
                for _ in range(3)
            ]
        )

        self.second_perf = nn.ModuleList(
            [
                nn.Linear(self.metadata["d_embd_performance"], d_model),
                PositionalEncoder(d_model, max_seq_len, dropout),
                TransformerEncoder(
                    d_model, n_layers, n_heads, d_hidden, max_seq_len, dropout
                ),
            ]
        )

        self.decoder = TransformerDecoder(
            d_model, n_layers, n_heads, d_hidden, max_seq_len, dropout
        )

        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.metadata["d_embd_performance"]),
            nn.Softmax(dim=-1),
        )

        max_mask = self._generate_square_subsequent_mask(max_seq_len)
        self.register_buffer("max_mask", max_mask)

    def config(self):
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_hidden": self.d_hidden,
            "max_seq_len": self.max_seq_len,
        }

    def _generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        melody,
        style,
        performance,
        decoder_input,
        second_perf=None,
        alpha=1,
    ):
        # Note output is performance shifted right 1
        key_padding_mask = None
        inputs = (melody, style, performance, decoder_input)

        # Pad to max_seq_len and update key_padding_mask
        if melody.size(1) != self.max_seq_len:
            pad = partial(
                nn.functional.pad, pad=(0, 0, 0, self.max_seq_len - melody.size(1))
            )
            key_padding_mask = torch.ones_like(melody).bool()
            key_padding_mask[:, : melody.size(1)] = False
            inputs = [pad(inp) for inp in inputs]

        inputs = [encoder(inp) for encoder, inp in zip(self.token_encoders, inputs)]
        inputs = [inp + encoder(inp) for encoder, inp in zip(self.pos_encoders, inputs)]

        # Pass melody, style, performance through encoders
        inputs[:3] = [encoder(inp) for encoder, inp in zip(self.encoders, inputs)]

        # Mean aggregate performance and sum with melody and style
        inputs[2] = torch.mean(inputs[2], dim=1, keepdim=True)

        if not self.training and alpha < 1 - 1e-5:
            # Pass second performance through encoder
            second_perf = self.second_perf[0](second_perf)
            second_perf = second_perf + self.second_perf[1](second_perf)
            second_perf = self.second_perf[2](second_perf)
            second_perf = torch.mean(second_perf, dim=1, keepdim=True)

            # Linearly interpolate between performance and second performance
            inputs[2] = alpha * inputs[2] + (1 - alpha) * second_perf

        encoder_output = sum(inputs[:3])
        output = self.decoder(
            inputs[-1], encoder_output, self.max_mask, key_padding_mask
        )

        output = self.output(output)
        return output

    def generate(
        self,
        melody,
        style,
        melody_perf,
        style_perf,
        alpha=0.5,
        max_len=10_000,
    ):
        generated = torch.zeros(self.metadata["d_embd_performance"]).to(self.device)
        generated[-2] = 1

        generated = generated.unsqueeze(0).unsqueeze(0)

        self.eval()
        for _ in tqdm(range(max_len), desc="Generating"):
            pred = self.forward(
                melody,
                style,
                melody_perf,
                generated[:, -self.max_seq_len :],
                style_perf,
                alpha=alpha,
            )[0]
            token = torch.multinomial(pred[-1], 1)
            encoded = nn.functional.one_hot(token, self.metadata["d_embd_performance"])
            generated = torch.cat((generated, encoded.unsqueeze(0)), dim=1)

            if token[0, -1, -1] == 1:
                break

        return generated

    def generate_from_midi(
        self,
        melody_midi: Midi,
        style_midi: Midi,
        alpha: float = 0.5,
        max_len: int = 10_000,
    ):
        process_midi(
            melody_midi, os.path.join("output", "tmp"), False, (Melody, Performance)
        )
        process_midi(
            style_midi, os.path.join("output", "tmp"), False, (Style, Performance)
        )

        melody_data = load_json(melody_midi.idx, "tmp", "output")
        style_data = load_json(style_midi.idx, "tmp", "output")

        melody, melody_perf, style, style_perf = (
            expand_encoding(data).to(self.device)
            for data in (
                melody_data[Melody.encoding_type],
                melody_data[Performance.encoding_type],
                style_data[Style.encoding_type],
                style_data[Performance.encoding_type],
            )
        )
        self.generate(melody, style, melody_perf, style_perf, alpha, max_len)
