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


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs
    ) -> None:
        """
        Create a new recurrent encoder.

        :param rnn_type: RNN type: `gru` or `lstm`.
        :param hidden_size: Size of each RNN.
        :param emb_size: Size of the word embeddings.
        :param num_layers: Number of encoder RNN layers.
        :param dropout:  Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param bidirectional: Use a bi-directional RNN.
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """

        super(RecurrentEncoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    # pylint: disable=invalid-name, unused-argument
    def _check_shapes_input_forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        :param embed_src: embedded source tokens
        :param src_length: source length
        :param mask: source mask
        """
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
        # assert mask.shape == embed_src.shape
        assert len(src_length.shape) == 1

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(
            embed_src=embed_src, src_length=src_length, mask=mask
        )

        # apply dropout to the rnn input
        embed_src = self.emb_dropout(embed_src)

        packed = pack_padded_sequence(embed_src, src_length, batch_first=True)
        output, hidden = self.rnn(packed)

        # pylint: disable=unused-variable
        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output, batch_first=True)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(
            self.rnn.num_layers,
            2 if self.rnn.bidirectional else 1,
            batch_size,
            self.rnn.hidden_size,
        )
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        # pylint: disable=no-member
        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden
        return output, hidden_concat

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)

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
