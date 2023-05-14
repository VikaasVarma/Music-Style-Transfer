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

import torch.nn as nn

from hyperparameters import (
    ATTENTION_HIDDEN_SIZE,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    NUM_HEADS,
    DIM_FF,
    DROPOUT_PROB,
)

from model.relative_position_layer import RelativePositionEncoderLayer
from model.relative_position_layer import RelativePositionDecoderLayer


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model=ATTENTION_HIDDEN_SIZE,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        d_hidden=DIM_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT_PROB,
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                RelativePositionEncoderLayer(
                    d_model, n_heads, d_hidden, max_seq_len, dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, key_padding_mask=None):
        for layer in self.encoder_layers:
            # Encoder does not mask
            x = layer(x, key_padding_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model=ATTENTION_HIDDEN_SIZE,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        d_hidden=DIM_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT_PROB,
    ):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [
                RelativePositionDecoderLayer(
                    d_model, n_heads, d_hidden, max_seq_len, dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, q, kv, mask, key_padding_mask=None):
        # kv comes from the encoder, q is the input to the decoder
        for layer in self.decoder_layers:
            q = layer(q, kv, mask, key_padding_mask)

        return q
