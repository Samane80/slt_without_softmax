# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadedAttention(nn.Module):
    """
    نسخه FP32 سازگار با کد اصلی signjoey/SLT
    اسم attributeها عمداً مثل نسخه کوانتیزه نگه داشته شده.
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1,
                 qkv_bias=False, focusing_factor: int = 2, kernel_size: int = 5,
                 causal: bool = False, is_self_attention: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = size // num_heads
        self.dim = size
        self.focusing_factor = focusing_factor
        self.causal = causal
        self.is_self_attention = is_self_attention

        # اسم attributeها دقیقاً مثل نسخه Quant
        self.q_layer = nn.Linear(size, size, bias=qkv_bias)
        self.k_layer = nn.Linear(size, size, bias=qkv_bias)
        self.v_layer = nn.Linear(size, size, bias=qkv_bias)
        self.output_layer = nn.Linear(size, size)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

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
            value = query   # مهم برای سازگاری با فراخوانی قدیمی

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

        # Key-padding mask
        key_valid_mask = None
        if mask is not None and not self.causal:
            key_valid_mask = mask
            if key_valid_mask.dim() == 3:
                key_valid_mask = key_valid_mask.squeeze(1).unsqueeze(1).unsqueeze(-1)
            key_valid_mask = key_valid_mask.to(k.dtype)
            k = k * key_valid_mask
            v = v * key_valid_mask

        if self.causal:
            kv = torch.einsum("bhnd,bhne->bhnde", k, v)
            kv_cumsum = torch.cumsum(kv, dim=2)
            context = torch.einsum("bhnd,bhnde->bhne", q, kv_cumsum)

            k_cumsum = torch.cumsum(k, dim=2)
            z = torch.einsum("bhnd,bhnd->bhn", q, k_cumsum).unsqueeze(-1)
        else:
            kv = torch.einsum("bhnd,bhne->bhde", k, v)
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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size: int, ff_size: int, dropout: float = 0.1):
        super().__init__()
        self.pwff_layer = nn.ModuleList([
            nn.Linear(input_size, ff_size),   # [0]
            nn.GELU(),                        # [1]
            nn.Dropout(dropout),              # [2]
            nn.Linear(ff_size, input_size),   # [3]
            nn.Dropout(dropout),              # [4]
        ])

    def forward(self, x):
        x = self.pwff_layer[0](x)
        x = self.pwff_layer[1](x)
        x = self.pwff_layer[2](x)
        x = self.pwff_layer[3](x)
        x = self.pwff_layer[4](x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1):
        super().__init__()
        self.size = size
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(
            num_heads=num_heads, size=size, dropout=dropout,
            focusing_factor=2, kernel_size=5, is_self_attention=True
        )
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o


class TransformerDecoderLayer(nn.Module):
    def __init__(self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1):
        super().__init__()
        self.size = size
        self.trg_trg_att = MultiHeadedAttention(
            num_heads=num_heads, size=size, dropout=dropout,
            focusing_factor=2, causal=True, is_self_attention=True
        )
        self.src_trg_att = MultiHeadedAttention(
            num_heads=num_heads, size=size, dropout=dropout,
            focusing_factor=2, is_self_attention=False
        )
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor = None, trg_mask: Tensor = None):
        # decoder self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(h1_norm, memory, memory, mask=src_mask)

        # feed-forward
        o = self.feed_forward(self.dropout(h2) + h1)
        return o


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