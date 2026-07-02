# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from signjoey.helpers import freeze_params
from signjoey.transformer_layers import TransformerEncoderLayer, PositionalEncoding


# pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    Base encoder class
    """

    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size=512, ff_size=2048, num_layers=8, num_heads=4,
                 dropout=0.1, emb_dropout=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=hidden_size, num_heads=num_heads, ff_size=ff_size, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(size=hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self._output_size = hidden_size

    @property
    def output_size(self):
        return self._output_size

    def forward(self, embed_src, src_length, mask):
        x = self.pe(embed_src)
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.layer_norm(x)
        return x, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )
