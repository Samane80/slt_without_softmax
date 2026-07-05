# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head "Focused Linear Attention" module, with a calibrate/infer
    mode for folding the (per-head) normalization constant into a fixed
    multiply-add at inference time.

    Softmax has been replaced end-to-end with the mechanism from
    "FLatten Transformer: Vision Transformer using Focused Linear
    Attention":

        Sim(Q, K) = phi_p(Q) phi_p(K)^T                 (Focused Function)
        O         = Sim(Q, K) V + DWC(V)                (rank restoration)

    Two pieces from the paper are implemented (both were missing from a
    prior draft of this file, which used a plain ReLU^2 feature map with
    no DWC term -- that is closer to vanilla Linear Attention and lacks
    both the "focus ability" and the "feature diversity" the FLatten
    paper specifically adds on top of it):

      1. `_focused_function`: phi_p(x) = (||x|| / ||x**p||) * x**p,
         applied after ReLU. This sharpens Q/K similarity so attention
         can form Softmax-like peaked distributions (needed for e.g.
         CTC-based gloss recognition, which relies on precise temporal
         localization).
      2. `_depthwise_conv`: the DWC rank-restoration term. Linear
         attention's implicit attention matrix has rank <= min(N, d),
         which homogenizes many rows of the map; DWC(V) restores a
         locally full-rank component at negligible extra cost.

    The `mode` machinery (train / calibrate / infer) is unrelated to
    accuracy -- it is a pure inference-time optimization that avoids a
    division per forward pass by folding a calibrated 1/denom constant
    into either a per-head multiply (`infer`) or directly into
    `output_layer`'s weights (`fold_norm_const_into_output_layer`).

    IMPORTANT: DWC(V) is an *additive* term that never goes through the
    denom division, so `fold_norm_const_into_output_layer` is only valid
    when `self_attention=False` (no DWC term is added). For self-attention
    layers, keep using `mode="infer"` (the per-head multiply happens
    *before* DWC is added), and do not call
    `fold_norm_const_into_output_layer` on them -- see the assertion in
    that method.
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
            fp(x) = (||x|| / ||x**p||) * x**p (FLatten Transformer paper).
        :param self_attention: whether q, k, v all come from the same
            sequence (encoder self-attention, decoder self-attention).
            The DWC term is only added in this case (see class docstring).
        :param dwc_kernel_size: kernel size of the depthwise convolution
            used for rank restoration. Only used if `self_attention`.
            Must be odd.
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
        self.dropout = nn.Dropout(dropout)

        # eps فقط برای پایداری عددی division در فاز train/calibrate
        self.eps = 1e-6

        # Focused Function power "p" (FLatten Transformer paper)
        self.focusing_factor = focusing_factor

        # mode یکی از 'train' / 'calibrate' / 'infer'
        self.mode = "train"

        # ثابت per-head که در فاز calibrate پر و بعداً fold میشه
        self.register_buffer("norm_const", torch.ones(num_heads))

        # آمار جمع‌آوری‌شده در فاز calibrate (per-head)
        self.register_buffer("_calib_sum", torch.zeros(num_heads))
        self._calib_count = 0

        # Rank-restoration term (DWC): only meaningful for self-attention,
        # since it is added directly to V and requires query length ==
        # key/value length.
        self.self_attention = self_attention
        self.dwc_kernel_size = dwc_kernel_size
        if self.self_attention:
            self.dwc = nn.Conv1d(
                in_channels=size,
                out_channels=size,
                kernel_size=dwc_kernel_size,
                groups=size,  # depthwise: one independent filter per channel
                bias=False,
                padding=0,  # padding applied manually in `_depthwise_conv`
            )
        else:
            self.dwc = None

    def _focused_function(self, x: Tensor) -> Tensor:
        """
        Focused Function fp(.) from the Focused Linear Attention paper:

            phi_p(x) = fp(ReLU(x))
            fp(x)    = (||x|| / ||x**p||) * x**p

        This replaces the plain `ReLU(x)^2` feature map: squaring alone
        has no mechanism to keep the *direction* of x sharp relative to
        its own norm, which is exactly what lets Softmax form peaked
        attention distributions. fp does that while staying non-negative
        (needed for a valid linear-attention kernel).
        """
        x = torch.relu(x)
        x_norm = x.norm(dim=-1, keepdim=True)
        x_pow = x.pow(self.focusing_factor)
        x_pow_norm = x_pow.norm(dim=-1, keepdim=True) + self.eps
        return x_norm * x_pow / x_pow_norm

    def _depthwise_conv(
        self, v: Tensor, causal: bool, key_valid: Tensor = None
    ) -> Tensor:
        """
        Rank-restoration term DWC(V). See class docstring / Eq.(10) of the
        FLatten paper. Causal self-attention gets left-only padding so no
        future position leaks into the past.

        :param v: value projections *before* head-splitting,
            shape (batch, seq_len, size)
        :param causal: whether to use causal (left-only) padding
        :param key_valid: optional bool/float mask, shape (batch, seq_len)
        """
        x = v.transpose(1, 2)  # (batch, size, seq_len) for nn.Conv1d

        if key_valid is not None:
            x = x * key_valid.unsqueeze(1).to(x.dtype)

        if causal:
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

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask. شکل [B, 1, M] برای padding-only (مثل
            encoder self-attn یا src-trg attention)، یا شکل [B, M, M] برای
            causal+padding (مثل decoder self-attn). در حالت causal به‌جای
            ماتریس کامل attention از cumulative sum استفاده میشه.
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

        # Focused Function mapping (replaces the plain ReLU^2 feature map;
        # no separate sqrt(head_size) pre-scale is needed since fp already
        # normalizes by the ratio of norms).
        q_phi = self._focused_function(q_heads)  # B,H,Mq,Dh
        k_phi = self._focused_function(k_heads)  # B,H,Mk,Dh

        is_causal = mask is not None and mask.dim() == 3 and mask.size(1) > 1

        key_valid = None
        if mask is not None:
            if is_causal:
                # فرض: causal mask با padding ترکیب شده و padding برای همه‌ی
                # query ها یکسانه (right-padding استاندارد) -> آخرین سطر
                # ماسک کامل‌ترین الگوی padding رو نشون می‌ده.
                key_valid = mask[:, -1, :]
            else:
                key_valid = mask.squeeze(1)
            key_mask = key_valid.unsqueeze(1).unsqueeze(-1)  # [B,1,Mk,1]
            k_phi = k_phi.masked_fill(~key_mask, 0.0)
            v_heads = v_heads.masked_fill(~key_mask, 0.0)

        if not is_causal:
            # non-causal: یک بار sum روی کل بعد کلید کافیه
            kv = torch.einsum("bhtd,bhte->bhde", k_phi, v_heads)
            k_sum = k_phi.sum(dim=2)

            numerator = torch.einsum("bhtd,bhde->bhte", q_phi, kv)
            denom = torch.einsum("bhtd,bhd->bht", q_phi, k_sum).unsqueeze(-1) + self.eps
        else:
            # causal: cumulative sum به‌جای sum کامل تا هر position فقط به
            # کلیدهای قبل از خودش (و خودش) نگاه کنه
            kv_terms = torch.einsum("bhtd,bhte->bhtde", k_phi, v_heads)
            kv_cumsum = torch.cumsum(kv_terms, dim=2)
            k_cumsum = torch.cumsum(k_phi, dim=2)

            numerator = torch.einsum("bhtd,bhtde->bhte", q_phi, kv_cumsum)
            denom = torch.einsum("bhtd,bhtd->bht", q_phi, k_cumsum).unsqueeze(-1) + self.eps

        if self.mode == "train":
            attention = numerator / denom

        elif self.mode == "calibrate":
            with torch.no_grad():
                inv = (1.0 / denom.squeeze(-1)).mean(dim=(0, 2))  # میانگین per-head
                self._calib_sum += inv * batch_size
                self._calib_count += batch_size
            attention = numerator / denom

        elif self.mode == "infer":
            # هیچ division‌ای اینجا نیست؛ ضرب در ثابت per-head calibrate‌شده
            attention = numerator * self.norm_const.view(1, num_heads, 1, 1)

        else:
            raise ValueError(f"mode ناشناخته: {self.mode}")

        attention = self.dropout(attention)

        # get context vector and reshape back to [B, M, D]
        context = (
            attention.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        # rank-restoration term: O = phi(Q)phi(K)^T V + DWC(V)
        # (added *after* the norm_const multiply above, and *before*
        # output_layer -- see the fold-safety note in `fold_norm_const_
        # into_output_layer`)
        if self.self_attention:
            context = context + self._depthwise_conv(v_proj, is_causal, key_valid)

        output = self.output_layer(context)

        return output

    def finalize_calibration(self):
        """میانگین 1/denom جمع‌آوری‌شده (per-head) رو به عنوان ثابت نهایی ذخیره می‌کنه."""
        assert self._calib_count > 0, "قبلش باید چند batch رو در mode='calibrate' رد کنی"
        self.norm_const.copy_(self._calib_sum / self._calib_count)
        self.mode = "infer"

    def fold_norm_const_into_output_layer(self):
        """
        ثابت‌های per-head calibrate‌شده رو مستقیم توی وزن output_layer ضرب
        می‌کنه تا در inference نهایی حتی نیاز به ضرب norm_const هم نباشه؛
        فقط یک GEMM ساده (mult+add) باقی می‌مونه.

        هشدار: این فولد فقط وقتی درسته که *همه‌ی* ورودیِ output_layer از
        مسیر تقسیم‌بر-denom (یعنی همون attention) اومده باشه. اگر
        self_attention=True باشه، یک ترم DWC(V) هم به‌صورت جمعی (نه از
        مسیر denom) به context اضافه شده، و ضرب‌کردن وزن‌های output_layer
        در norm_const به‌اشتباه روی سهم DWC هم اثر می‌ذاره. برای همین این
        متد فقط برای لایه‌های cross-attention (self_attention=False)
        مجازه.
        """
        assert self.mode == "infer", "اول finalize_calibration() رو صدا بزن"
        assert not self.self_attention, (
            "fold_norm_const_into_output_layer فقط برای لایه‌های "
            "cross-attention (self_attention=False) امنه، چون DWC یک ترم "
            "جمعی است که از مسیر denom رد نمی‌شه. برای لایه‌های self-attention "
            "در mode='infer' بمون (ضرب per-head قبل از DWC انجام می‌شه)."
        )
        with torch.no_grad():
            for h in range(self.num_heads):
                col_slice = slice(h * self.head_size, (h + 1) * self.head_size)
                self.output_layer.weight[:, col_slice] *= self.norm_const[h].item()
        self.norm_const.fill_(1.0)


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