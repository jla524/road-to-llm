"""
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    block_size = 1024
    vocab_size = 50304
    num_layers = 12
    num_heads = 12
    num_embeddings = 768
    dropout = 0.0
    bias = True


class LayerNorm(nn.Module):
    def __init__(self, num_dims, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_dims))
        self.bias = nn.Parameter(torch.zeros(num_dims)) if bias else None

    def __call__(self, x):
        x = F.layer_norm(x, self.weight.shape, weight=self.weight, bias=self.bias, eps=1e-5)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attnention = nn.Linear(config.num_embeddings, 3 * config.num_embeddings, bias=config.bias)
        self.projection = nn.Linear(config.num_embeddings, config.num_embeddings, bias=config.bias)
        self.num_heads = config.num_heads
        self.num_embeddings = config.num_embeddings
        self.dropout = config.dropout

    def __call__(self, x):
        batch_size, sequence_length, num_dimensions = x.size()
        query, key, value = self.attention(x).split(self.num_embeddings, dim=2)
        query = query.view(batch_size, sequence_length, num_dimensions // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, sequence_length, num_dimensions // self.num_heads).transpose(1, 2)
        value = value.view(batch_size, sequence_length, num_dimensions // self.num_heads).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous.view(batch_size, sequence_length, num_dimensions)
        y = self.projection(y).dropout(self.dropout)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fully_connected = nn.Linear(config.num_embeddings, 4 * config.num_embeddings, bias=config.bias)
        self.gelu = nn.GELU()
        self.projection = nn.Linear(4 * config.num_embeddings, config.num_embeddings, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.gelu(self.fully_connected(x))
        x = self.dropout(self.projection(x))
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.num_embeddings, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.num_embeddings, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln1(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.num_embeddings),
            "wpe": nn.Embedding(config.vocab_size, config.num_embeddings),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            "ln": LayerNorm(config.num_embeddings, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.num_embeddings, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.__init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("projection.weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02/math.sqrt(2*config.num_layers))
        print(f"number of parameters {self.get_num_params() / 1e6:.2f}M")

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.norm_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zero_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.transfromer.wpe.weight.numel()
        return num_params

    def __call__(self, idx, targets=None):
        ...
