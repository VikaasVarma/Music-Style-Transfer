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

import torch
import torch.nn as nn
import math

from hyperparameters import MAX_SEQ_LEN, DROPOUT_PROB


class PositionalEncoder(nn.Module):
    # Cosine Sine Positional Encoder

    def __init__(self, d_model, max_seq_len=MAX_SEQ_LEN, dropout=DROPOUT_PROB):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position_encoding = torch.zeros(1, max_seq_len, d_model)

        pos = torch.arange(max_seq_len).unsqueeze(1)
        scaling = torch.exp(torch.arange(0, d_model) * (-math.log(1e4) / d_model))

        position_encoding[0, :, 0::2] = torch.sin(pos * scaling[0::2])
        position_encoding[0, :, 1::2] = torch.cos(pos * scaling[0::2])
        self.register_buffer("position_encoding", position_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.position_encoding[:, : x.size(1)]
        return self.dropout(x)
