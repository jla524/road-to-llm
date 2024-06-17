"""
https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
"""
import math
import torch
from torch import nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, input_dims, max_sequence_length):
        super().__init__()
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dims, 2).float() * -(math.log(10000.0) / input_dims))
        pe = torch.zeros(max_sequence_length, input_dims)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # adds a buffer to the module

    def __call__(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dims, num_heads):
        super().__init__()
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.k = input_dims // num_heads
        self.wq = nn.Linear(input_dims, input_dims)
        self.wk = nn.Linear(input_dims, input_dims)
        self.wv = nn.Linear(input_dims, input_dims)
        self.wo = nn.Linear(input_dims, input_dims)

    def split_heads(self, x):
        return rearrange(x, "b s (h k) -> b h s k", h=self.num_heads, k=self.k)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, v)

    def combined_heads(self, x):
        return rearrange(x, "b h s k -> b s (h k)")

    def __call__(self, q, k, v, mask=None):
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        attn_output = self.scaled_dot_product_attention(q, k, v, mask=mask)
        return self.wo(self.combined_heads(attn_output))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super().__init__()
        self.l1 = nn.Linear(input_dims, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, input_dims)

    def __call__(self, x):
        x = self.l1(x).relu()
        return self.l2(x)


class EncoderLayer(nn.Module):
    def __init__(self, input_dims, num_heads, hidden_dims, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dims, num_heads)
        self.feed_forward = PositionWiseFeedForward(input_dims, hidden_dims)
        self.norm1 = nn.LayerNorm(input_dims)
        self.norm2 = nn.LayerNorm(input_dims)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, input_dims, num_heads, hidden_dims, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dims, num_heads)
        self.cross_attn = MultiHeadAttention(input_dims, num_heads)
        self.feed_forward = PositionWiseFeedForward(input_dims, hidden_dims)
        self.norm1 = nn.LayerNorm(input_dims)
        self.norm2 = nn.LayerNorm(input_dims)
        self.norm3 = nn.LayerNorm(input_dims)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, encoder_output, source_mask, target_mask):
        attn_output = self.self_attn(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, encoder_output, encoder_output, source_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dims, num_heads, num_layers, ff_dims, max_sequence_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, hidden_dims)
        self.positional_encoding = PositionalEncoding(hidden_dims, max_sequence_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dims, num_heads, ff_dims, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dims, num_heads, ff_dims, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dims, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3)
        sequence_length = target.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, sequence_length, sequence_length), diagonal=1)).bool()
        target_mask = target_mask & nopeak_mask
        return source_mask, target_mask

    def __call__(self, source):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        encoder_output = self.dropout(self.positional_encoding(self.encoder_embedding(source)))
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, source_mask)
        return self.fc(encoder_output)
