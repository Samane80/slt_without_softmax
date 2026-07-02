# coding: utf-8

"""
Various decoders
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from signjoey.attention import BahdanauAttention, LuongAttention
from signjoey.encoders import Encoder
from signjoey.helpers import freeze_params, subsequent_mask
from signjoey.transformer_layers import PositionalEncoding, TransformerDecoderLayer


# pylint: disable=abstract-method
class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=4, num_heads=8, hidden_size=512, ff_size=2048,
                 dropout=0.1, emb_dropout=0.1, vocab_size=1, **kwargs):
        super().__init__()
        self._hidden_size = hidden_size
        self._output_size = vocab_size
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.pe = PositionalEncoding(size=hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def output_size(self):
        return self._output_size

    def forward(self, trg_embed, encoder_output, src_mask, trg_mask, **kwargs):
        x = self.pe(trg_embed)
        x = self.emb_dropout(x)
        trg_mask = trg_mask & subsequent_mask(trg_embed.size(1)).type_as(trg_mask)
        for layer in self.layers:
            x = layer(x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)
        x = self.layer_norm(x)
        output = self.output_layer(x)
        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].trg_trg_att.num_heads,
        )
