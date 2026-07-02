# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0,
                 qkv_bias=False, focusing_factor: int = 2, kernel_size: int = 5,
                 causal: bool = False, is_self_attention: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = dim // num_heads
        self.dim = dim
        self.focusing_factor = focusing_factor
        self.causal = causal
        self.is_self_attention = is_self_attention

        self.q_layer = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_layer = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_layer = nn.Linear(dim, dim, bias=qkv_bias)
        self.output_layer = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # همون کرنل (1,k) روی محور توالی — نه grid دوبعدی مصنوعی
        self.dwc = nn.Conv2d(
            self.head_size, self.head_size, kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2), groups=self.head_size, bias=True
        )

    def _focused_map(self, x):
        x = F.relu(x) + 1e-6
        xp = x * x if self.focusing_factor == 2 else x * x * x
        x_norm = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + 1e-8
        xp_norm = torch.max(torch.abs(xp), dim=-1, keepdim=True)[0] + 1e-8
        scale = x_norm / (xp_norm + 1e-6)
        return scale * xp

    def forward(self, query, key=None, value=None, mask=None):
        if key is None:
            key = query
        if value is None:
            value = key

        B, Nq, C = query.shape
        Nk = key.shape[1]

        q = self.q_layer(query)
        k = self.k_layer(key)
        v = self.v_layer(value)

        q = q.reshape(B, Nq, self.num_heads, self.head_size).transpose(1, 2)
        k = k.reshape(B, Nk, self.num_heads, self.head_size).transpose(1, 2)
        v = v.reshape(B, Nk, self.num_heads, self.head_size).transpose(1, 2)

        q = self._focused_map(q)
        k = self._focused_map(k)

        # ── Key-padding mask (فقط برای حالت غیر-causal) ────────────────
        key_valid_mask = None
        if mask is not None and not self.causal:
            key_valid_mask = mask
            if key_valid_mask.dim() == 3:
                key_valid_mask = key_valid_mask.squeeze(1).unsqueeze(1).unsqueeze(-1)
            key_valid_mask = key_valid_mask.to(k.dtype)
            k = k * key_valid_mask
            v = v * key_valid_mask

        if self.causal:
            # causal linear attention با prefix-sum روی محور توالی
            kv = torch.einsum("bhnd,bhne->bhnde", k, v)
            kv_cumsum = torch.cumsum(kv, dim=2)
            context = torch.einsum("bhnd,bhnde->bhne", q, kv_cumsum)

            k_cumsum = torch.cumsum(k, dim=2)
            z = torch.einsum("bhnd,bhnd->bhn", q, k_cumsum).unsqueeze(-1)
        else:
            kv = torch.einsum("bhnd,bhne->bhde", k, v)  # جمع سراسری روی Nk
            context = torch.einsum("bhnd,bhde->bhne", q, kv)

            if key_valid_mask is not None:
                valid_counts = key_valid_mask.sum(dim=2, keepdim=True).clamp(min=1)
                k_mean = (k.sum(dim=-2, keepdim=True) / valid_counts).squeeze(-2)
            else:
                k_mean = k.mean(dim=-2)
            z = torch.einsum("bhnd,bhd->bhn", q, k_mean).unsqueeze(-1)

        z = 1.0 / (z + 1e-6)
        context = context * z

        if self.is_self_attention:
            v_1d = v.permute(0, 1, 3, 2).reshape(
                B * self.num_heads, self.head_size, 1, Nk
            )
            dwc_out = self.dwc(v_1d)
            dwc_out = dwc_out.reshape(
                B, self.num_heads, self.head_size, Nk
            ).permute(0, 1, 3, 2)
            context = context + dwc_out

        context = context.transpose(1, 2).contiguous().view(B, Nq, C)
        context = self.proj_drop(context)
        output = self.output_layer(context)
        return output



# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.pwff_layer = nn.ModuleList([
            nn.Linear(in_features, hidden_features),   # [0]
            nn.GELU(),                                  # [1]
            nn.Dropout(drop),                            # [2]
            nn.Linear(hidden_features, out_features),   # [3]
            nn.Dropout(drop),                            # [4]
        ])

    def forward(self, x):
        x = self.pwff_layer[0](x)
        x = self.pwff_layer[1](x)
        x = self.pwff_layer[2](x)
        x = self.pwff_layer[3](x)
        x = self.pwff_layer[4](x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(self, dim: int, num_heads: int, ff_size: int,
                 dropout: float = 0.1, attn_drop: float = 0.1):
        super().__init__()
        self.size = dim
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(
            dim=dim, num_heads=num_heads, dropout=attn_drop,
            focusing_factor=2, kernel_size=5, is_self_attention=True,
        )
        self.feed_forward = PositionwiseFeedForward(
            in_features=dim, hidden_features=ff_size, drop=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x_norm = self.layer_norm(x)
        x_att = self.src_src_att(x_norm, mask=mask)
        x = residual + self.dropout(x_att)

        residual = x
        x_ff = self.feed_forward(x)
        x = residual + self.dropout(x_ff)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, size: int, ff_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.size = size
        self.trg_trg_att = MultiHeadedAttention(
            dim=size, num_heads=num_heads, dropout=dropout,
            focusing_factor=2, causal=True, is_self_attention=True,
        )
        self.src_trg_att = MultiHeadedAttention(
            dim=size, num_heads=num_heads, dropout=dropout,
            focusing_factor=2, is_self_attention=False,
        )
        self.feed_forward = PositionwiseFeedForward(
            in_features=size, hidden_features=ff_size, drop=dropout
        )
        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, trg_mask=None):
        residual = x
        x_norm = self.x_layer_norm(x)
        x_att = self.trg_trg_att(query=x_norm, key=x_norm, value=x_norm, mask=trg_mask)
        x = residual + self.dropout(x_att)

        residual = x
        x_norm = self.dec_layer_norm(x)
        x_att = self.src_trg_att(query=x_norm, key=memory, value=memory, mask=src_mask)
        x = residual + self.dropout(x_att)

        residual = x
        x_ff = self.feed_forward(x)
        x = residual + self.dropout(x_ff)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, size: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, emb):
        return emb + self.pe[:, :emb.size(1)]



