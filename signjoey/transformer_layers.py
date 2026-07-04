# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head "Focused Linear Attention" module.

    This started as the Multi-Head (Softmax) Attention from
    "Attention is All You Need" (implementation modified from
    OpenNMT-py, https://github.com/OpenNMT/OpenNMT-py). The Softmax
    similarity has been replaced end-to-end with the mechanism proposed
    in "FLatten Transformer: Vision Transformer using Focused Linear
    Attention":

        Sim(Q, K) = phi_p(Q) phi_p(K)^T                                (Eq. 5/6)
        O         = Sim(Q, K) V + DWC(V)                               (Eq. 10/12)

    Two things are implemented, matching the paper:
      1. The "Focused Function" phi_p, a norm-based re-weighting of Q/K
         that restores a Softmax-like sharp attention distribution
         (`_focused_function`).
      2. True linear-complexity attention (O(N) instead of O(N^2)) via
         the associative reordering Q(K^T V) instead of (QK^T)V
         (`_parallel_linear_attention` for non-causal / globally-masked
         attention, `_causal_linear_attention` with a running state for
         causally-masked self-attention), plus the depthwise-convolution
         (DWC) rank-restoration term applied to V for self-attention
         layers (`_depthwise_conv`).
    """

    def __init__(
        self,
        num_heads: int,
        size: int,
        dropout: float = 0.1,
        focusing_factor: int = 3,
        self_attention: bool = True,
        dwc_kernel_size: int = 5,
    ):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        :param focusing_factor: power "p" used in the Focused Function
            fp(x) = (||x|| / ||x**p||) * x**p, proposed in "FLatten
            Transformer: Vision Transformer using Focused Linear
            Attention". This replaces the original Softmax similarity
            with a kernel-based (linear-attention-style) similarity.
        :param self_attention: whether query, key and value all come
            from the same sequence (encoder self-attention, decoder
            self-attention). The DWC rank-restoration term of Eq.(10)
            only makes sense in this case, since it is added directly
            to V and therefore requires the query length and the
            key/value length to match. Set to False for cross-attention
            (e.g. the decoder attending to the encoder output), where
            no DWC term is added.
        :param dwc_kernel_size: kernel size of the depthwise convolution
            (DWC) used for rank restoration, applied along the sequence
            dimension. Only used if `self_attention` is True. Must be odd.
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0
        assert dwc_kernel_size % 2 == 1, "dwc_kernel_size must be odd"

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Focused Linear Attention: replaces the Softmax similarity
        # Sim(Q, K) = exp(QK^T / sqrt(d)) with a kernel similarity
        # Sim(Q, K) = phi_p(Q) phi_p(K)^T, where phi_p sharpens the
        # direction of Q/K using a norm-based "focused function"
        # while preserving their original norm.
        self.focusing_factor = focusing_factor
        self.eps = 1e-6

        # Rank-restoration term (Eq. 10/12): O = Sim(Q,K)V + DWC(V).
        # Only meaningful for self-attention (query/key/value share the
        # same sequence length), since DWC(V) is added directly to the
        # attention output.
        self.self_attention = self_attention
        self.dwc_kernel_size = dwc_kernel_size
        if self.self_attention:
            self.dwc = nn.Conv1d(
                in_channels=size,
                out_channels=size,
                kernel_size=dwc_kernel_size,
                groups=size,  # depthwise: one independent filter per channel
                bias=False,
                padding=0,  # padding is applied manually in `_depthwise_conv`
            )
        else:
            self.dwc = None

    def _focused_function(self, x: Tensor) -> Tensor:
        """
        Focused Function fp(.) from the Focused Linear Attention paper:

            phi_p(x) = fp(ReLU(x))
            fp(x)    = (||x|| / ||x**p||) * x**p

        ReLU ensures non-negativity (needed for a valid linear-attention
        kernel and for the denominator in the normalization below).
        The mapping only adjusts the *direction* of x: the norm of the
        output equals the norm of ReLU(x), i.e. ||fp(x)|| = ||ReLU(x)||.

        :param x: input tensor, e.g. projected queries or keys
        :return: direction-sharpened tensor with the same shape as x
        """
        x = torch.relu(x)
        x_norm = x.norm(dim=-1, keepdim=True)
        x_pow = x.pow(self.focusing_factor)
        x_pow_norm = x_pow.norm(dim=-1, keepdim=True) + self.eps
        return x_norm * x_pow / x_pow_norm

    def _parallel_linear_attention(
        self, q: Tensor, k: Tensor, v: Tensor, key_valid: Tensor = None
    ) -> Tensor:
        """
        True O(N) linear attention for non-causal (or unmasked)
        attention, i.e. attention where every query is allowed to see
        the same set of keys (only padding may be masked out, not
        position). Uses the associative-reordering trick of Eq.(3)/(4):

            O = phi(Q) (phi(K)^T V) / phi(Q) (phi(K)^T 1)

        instead of the O(N^2) computation (phi(Q)phi(K)^T) V.

        :param q: focused queries, shape (batch, heads, query_len, head_size)
        :param k: focused keys,    shape (batch, heads, key_len, head_size)
        :param v: values,          shape (batch, heads, key_len, head_size)
        :param key_valid: optional bool/float mask, shape (batch, key_len),
            True/1 at valid (non-padded) key positions.
        :return: context vectors, shape (batch, heads, query_len, head_size)
        """
        if key_valid is not None:
            # zero out padded keys so they don't contribute to the sums below
            key_valid = key_valid.unsqueeze(1).unsqueeze(-1).to(k.dtype)
            k = k * key_valid

        # kv: (batch, heads, head_size, head_size) = sum_j k_j (outer) v_j
        kv = torch.matmul(k.transpose(-2, -1), v)
        # k_sum: (batch, heads, head_size) = sum_j k_j
        k_sum = k.sum(dim=-2)

        numerator = torch.matmul(q, kv)  # (batch, heads, query_len, head_size)
        denominator = torch.matmul(q, k_sum.unsqueeze(-1)) + self.eps
        # denominator: (batch, heads, query_len, 1)

        return numerator / denominator

    def _causal_linear_attention(
        self, q: Tensor, k: Tensor, v: Tensor, key_valid: Tensor = None
    ) -> Tensor:
        """
        True O(N) linear attention for causally-masked self-attention
        (e.g. the decoder), where query i may only attend to keys j <= i.
        Implemented with a running ("RNN-style") state so that we never
        materialize the full (query_len x key_len) attention matrix nor
        an (N x head_size x head_size) intermediate tensor:

            kv_state_i = sum_{j<=i} k_j (outer) v_j
            k_state_i  = sum_{j<=i} k_j
            O_i        = phi(q_i) kv_state_i / phi(q_i) k_state_i

        :param q: focused queries, shape (batch, heads, seq_len, head_size)
        :param k: focused keys,    shape (batch, heads, seq_len, head_size)
        :param v: values,          shape (batch, heads, seq_len, head_size)
        :param key_valid: optional bool/float mask, shape (batch, seq_len),
            True/1 at valid (non-padded) key positions.
        :return: context vectors, shape (batch, heads, seq_len, head_size)
        """
        batch_size, num_heads, seq_len, head_size = q.shape

        if key_valid is not None:
            key_valid = key_valid.unsqueeze(1).unsqueeze(-1).to(k.dtype)
            k = k * key_valid

        kv_state = q.new_zeros(batch_size, num_heads, head_size, head_size)
        k_state = q.new_zeros(batch_size, num_heads, head_size)
        outputs = []
        for t in range(seq_len):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            # accumulate running sums (this is the O(N) "recurrence")
            kv_state = kv_state + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            k_state = k_state + k_t

            q_t = q[:, :, t, :]
            numerator = torch.einsum("bhd,bhde->bhe", q_t, kv_state)
            denominator = (
                torch.einsum("bhd,bhd->bh", q_t, k_state).unsqueeze(-1) + self.eps
            )
            outputs.append(numerator / denominator)

        return torch.stack(outputs, dim=2)

    def _depthwise_conv(
        self, v: Tensor, causal: bool, key_valid: Tensor = None
    ) -> Tensor:
        """
        Rank-restoration term DWC(V) from Eq.(10)/(12). Linear attention's
        equivalent attention matrix is bounded in rank by min(N, head_size),
        which homogenizes many rows of the (implicit) attention map. Adding
        a lightweight depthwise convolution over V restores a locally
        full-rank component, at little extra computation.

        For causal self-attention (the decoder), the convolution is
        left-padded only, so no future position ever leaks into the past
        (autoregressive property preserved).

        :param v: value projections *before* head-splitting,
            shape (batch, seq_len, size)
        :param causal: whether to use causal (left-only) padding
        :param key_valid: optional bool/float mask, shape (batch, seq_len)
        :return: DWC(V), shape (batch, seq_len, size)
        """
        x = v.transpose(1, 2)  # (batch, size, seq_len) for nn.Conv1d

        if key_valid is not None:
            x = x * key_valid.unsqueeze(1).to(x.dtype)

        if causal:
            # pad only on the left: query i must not see keys j > i
            x = F.pad(x, (self.dwc_kernel_size - 1, 0))
        else:
            pad = (self.dwc_kernel_size - 1) // 2
            x = F.pad(x, (pad, pad))

        x = self.dwc(x)  # depthwise conv, output length == input seq_len

        return x.transpose(1, 2)  # back to (batch, seq_len, size)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed Focused Linear Attention:

            O = phi_p(Q) phi_p(K)^T V + DWC(V)

        with true O(N) complexity (no (query_len x key_len) attention
        matrix is ever materialized).

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask. Either [B, 1, M] (same mask for every
            query, e.g. padding-only masks used for encoder self-attention
            and for cross-attention), or [B, T, M] with T == M (a mask
            that varies per query position, e.g. the causal + padding
            mask used for decoder self-attention).
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k_proj = self.k_layer(k)
        v_proj = self.v_layer(v)  # keep un-split for the DWC term below
        q_proj = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k_heads = k_proj.view(batch_size, -1, num_heads, self.head_size).transpose(
            1, 2
        )
        v_heads = v_proj.view(batch_size, -1, num_heads, self.head_size).transpose(
            1, 2
        )
        q_heads = q_proj.view(batch_size, -1, num_heads, self.head_size).transpose(
            1, 2
        )

        # compute the Focused Function mapping (replaces the softmax scaling)
        q_heads = self._focused_function(q_heads)
        k_heads = self._focused_function(k_heads)

        # figure out mask semantics:
        # - mask.size(-2) == 1  -> same mask for every query (padding only)
        # - otherwise           -> per-query mask (causal + padding),
        #                          only used for masked self-attention
        causal = False
        key_valid = None
        if mask is not None:
            if mask.size(-2) == 1:
                key_valid = mask.squeeze(-2)  # (batch, key_len)
            else:
                causal = True
                # validity of each key regardless of the causal constraint:
                # for the last (right-most) query, causal never masks
                # anything, so only padding remains.
                key_valid = mask[:, -1, :]  # (batch, key_len)

        # true O(N) linear attention (no NxN attention matrix)
        if causal:
            context = self._causal_linear_attention(
                q_heads, k_heads, v_heads, key_valid
            )
        else:
            context = self._parallel_linear_attention(
                q_heads, k_heads, v_heads, key_valid
            )

        # get context vector and reshape back to [B, M, D]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )
        context = self.dropout(context)

        # rank-restoration term: O = phi(Q)phi(K)^T V + DWC(V), Eq.(10)/(12)
        # (only defined for self-attention, see docstring of __init__)
        if self.self_attention:
            context = context + self._depthwise_conv(v_proj, causal, key_valid)

        output = self.output_layer(context)

        return output


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(
            num_heads, size, dropout=dropout, self_attention=True
        )
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(
            num_heads, size, dropout=dropout, self_attention=True
        )
        self.src_trg_att = MultiHeadedAttention(
            num_heads, size, dropout=dropout, self_attention=False
        )

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o