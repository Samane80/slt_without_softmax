import math
import torch

from torch import nn, Tensor
import torch.nn.functional as F
# from signjoey.helpers import freeze_params


def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "celu":
        return nn.CELU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softplus":
        return nn.Softplus()
    elif activation_type == "softshrink":
        return nn.Softshrink()
    elif activation_type == "softsign":
        return nn.Softsign()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("Unknown activation type {}".format(activation_type))


class MaskedNorm(nn.Module):
    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")
        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape([-1, self.num_features])
            normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            normed = self.norm(reshaped)
            return normed.reshape([x.shape[0], -1, self.num_features])


class Embeddings(nn.Module):
    def __init__(self, embedding_dim=64, num_heads=8, scale=False, scale_factor=None,
                 norm_type=None, activation_type=None, vocab_size=0, padding_idx=1, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.lut = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.lut.weight, mean=0, std=embedding_dim ** -0.5)

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(norm_type, num_heads, embedding_dim)

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            self.scale_factor = scale_factor if scale_factor else math.sqrt(embedding_dim)

    def forward(self, x: Tensor, mask: Tensor = None):
        x = self.lut(x)
        if self.scale:
            x = x * self.scale_factor
        if self.norm_type:
            x = self.norm(x, mask)
        if self.activation_type:
            x = self.activation(x)
        return x


class SpatialEmbeddings(nn.Module):
    def __init__(self, embedding_dim, input_size, num_heads, norm_type=None,
                 activation_type=None, scale=False, scale_factor=None, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size

        self.ln = nn.Linear(input_size, embedding_dim)

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(norm_type, num_heads, embedding_dim)

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            self.scale_factor = scale_factor if scale_factor else math.sqrt(embedding_dim)

    def forward(self, x: Tensor, mask: Tensor):
        x = self.ln(x)
        if self.norm_type:
            x = self.norm(x, mask)
        if self.activation_type:
            x = self.activation(x)
        if self.scale:
            x = x * self.scale_factor
        return x