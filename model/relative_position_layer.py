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

import numpy as np
import torch
import torch.nn as nn
import math

from hyperparameters import DIM_FF, NUM_HEADS, MAX_SEQ_LEN, DROPOUT_PROB


class RelativePositionEncoderLayer(nn.Module):
    # Pytorch implementation of memory efficient transformer encoder
    # layer with relative position encoding

    def __init__(
        self,
        d_model,
        n_head=NUM_HEADS,
        dim_ff=DIM_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT_PROB,
    ):
        super().__init__()

        self.self_attention = RPEMultiHeadAttention(
            d_model, n_head, max_seq_len, dropout
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        self.norms = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout(dropout), nn.LayerNorm(d_model)),
                nn.Sequential(nn.Dropout(dropout), nn.LayerNorm(d_model)),
            ]
        )

    def forward(self, x, key_padding_mask=None):
        # Note in encoders query, key, value are all the same and there is no mask
        # Norm first empirically produces better results
        x = self.norms[0](x)
        x = x + self.self_attention(x, x, x, None, key_padding_mask)
        x = self.norms[1](x)
        x = x + self.ff(x)
        return x


class RelativePositionDecoderLayer(nn.Module):
    # Pytorch implementation of memory efficient transformer encoder
    # layer with relative position encoding

    def __init__(
        self,
        d_model,
        n_head=NUM_HEADS,
        dim_ff=DIM_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT_PROB,
    ):
        super().__init__()

        self.self_attention, self.multi_head_attention = (
            RPEMultiHeadAttention(d_model, n_head, max_seq_len, dropout)
            for _ in range(2)
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        self.norms = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout(dropout), nn.LayerNorm(d_model)),
                nn.Sequential(nn.Dropout(dropout), nn.LayerNorm(d_model)),
                nn.Sequential(nn.Dropout(dropout), nn.LayerNorm(d_model)),
            ]
        )

    def forward(self, q, kv, mask=None, key_padding_mask=None):
        # Note in this task key and value are always the same and mask is only used for self attention
        # Norm first empirically produces better results
        q = self.norms[0](q)
        q = q + self.self_attention(q, q, q, mask, key_padding_mask)
        q = self.norms[1](q)
        x = q + self.multi_head_attention(q, kv, kv, None, key_padding_mask)
        x = self.norms[2](q)
        x = x + self.ff(x)
        return x


class RPEMultiHeadAttention(nn.Module):
    # Pytorch implementation of memory efficient Multi-head attention with relative position encoding
    # Specialized for use-case of query, key, value all being the same (because we are using autoencoder)

    def __init__(
        self, d_model, n_head=NUM_HEADS, max_seq_len=MAX_SEQ_LEN, dropout=DROPOUT_PROB
    ):
        super().__init__()
        assert (
            d_model % n_head == 0
        ), f"Embedding dimension must be divisible by number of heads {d_model} {n_head}"

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.max_seq_len = max_seq_len

        self.q_proj, self.k_proj, self.v_proj = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model))
            for _ in range(3)
        )
        self.attn_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model))

        # RPE Embedding matrix
        self.Er = nn.Parameter(torch.rand(max_seq_len, self.d_head), requires_grad=True)
        mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).float().flip(0)
        self.register_buffer("mask", mask)

    def _skew(self, qer):
        qer = self.mask * qer
        qer = nn.functional.pad(qer, (1, 0))  # B * N, S, S + 1
        qer = torch.reshape(qer, (qer.shape[0], qer.shape[2], qer.shape[1]))
        return qer[:, 1:, :]

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        q = q * 1 / math.sqrt(self.d_head)

        # Let B = batch_size, S = seq_len, N = n_head, E = d_model
        batch_size, seq_len, _ = q.size()  # B, S, E

        q, k, v = (
            inp.transpose(0, 1)
            .contiguous()
            .view(seq_len, batch_size * self.n_head, -1)
            .transpose(0, 1)
            for inp in (q, k, v)
        )  # B * N, S, E / N

        attention = torch.einsum("bse,bte->bst", q, k)  # B * N, S, S
        qer = torch.einsum("bse,te->bst", q, self.Er)  # B * N, S, S
        srel = self._skew(qer)  # B * N, S, S

        attention = attention + srel
        if mask is not None:
            attention += mask.unsqueeze(0)  # mask: 1, S, S
        if key_padding_mask is not None:
            attention = attention.view(batch_size, self.n_head, seq_len, seq_len)
            attention = attention.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # B, 1, 1, S
                -np.inf,
            )
            attention = attention.view(batch_size * self.n_head, seq_len, seq_len)

        attention = nn.functional.softmax(attention, dim=-1)  # B * N, S, S

        output = torch.bmm(attention, v)  # B * N, S, E / N
        output = (
            output.transpose(0, 1)
            .contiguous()
            .view(seq_len, batch_size, self.d_model)
            .transpose(0, 1)
        )

        output = self.attn_proj(output)  # B, S, E
        return output
