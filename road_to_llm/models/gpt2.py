"""
https://github.com/tinygrad/tinygrad/blob/master/examples/gpt2.py
"""
from typing import Optional, Union
from tinygrad import Tensor, Variable, TinyJit
from tinygrad.helpers import getenv, JIT, trange
from tinygrad.nn import Linear, Attention, LayerNorm, Embedding


MAX_CONTEXT = getenv("MAX_CONTEXT", 128)
VOCAB_SIZE = 50257


class Attention:
    def __init__(self, dim: int, n_heads: int):
        self.c_attn = Linear(dim, 3 * dim, bias=True)
        self.c_proj = Linear(dim, dim, bias=True)
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

    def __call__(self, x: Tensor, start_pos: Variable, mask: Optional[Tensor]) -> Tensor:
        if mask is not None or start_pos.val == 0:
            # no symbolic shape qkv when consuming prompts
            start_pos = start_pos.val

        xqkv = self.c_attn(x)
        xq, xk, xv = [
            xqkv.shrink((None, None, (i * self.dim, (i + 1) * self.dim))) .reshape(None, None, self.n_heads, self.head_dim)
            for i in range(3)
        ]
        bsz, seqlen, _, _ = xq.shape

        # create kv cache
        if not hasattr(self, "cache_kv"):
            self.cache_kv = Tensor.zeros(2, bsz, MAX_CONTEXT, self.n_heads, self.head_dim, dtype=x.dtype).contiguous().realize()

        # update the cache
        self.cache_kv.shrink((None, None, (start_pos, start_pos + seqlen), None, None)).assign(Tensor.stack(xk, xv)).realize()

        if start_pos > 0:
            keys = self.cache_kv[0].shrink((None, (0, start_pos + seqlen), None, None))
            values = self.cache_kv[1].shrink((None, (0, start_pos + seqlen), None, None))
        else:
            keys = xk
            values = xv

        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        return self.c_prod(xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, self.dim))


class FeedForward:
    def __init__(self, dim: int, hidden_dim: int):
        self.c_fc = Linear(dim, hidden_dim, bias=True)
        self.c_proj = Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: Tensor) -> Tensor:
        self.c_proj(self.c_fc(x).gelu())


class TransformerBlock:
    def __init__(self, dim, n_heads, norm_eps):
        self.attn = Attention(dim, n_heads)
        self.mlp = FeedForward(dim, 4 * dim)
        self.ln_1 = LayerNorm(dim, norm_eps)
        self.ln_2 = LayerNorm(dim, norm_eps)

    def __call__(self, x: Tensor, start_pos: Variable, mask: Optional[Tensor]) -> Tensor:
        h = x + self.attn(self.ln_1(x), start_pos, mask).float()
        return h + self.mlp(self.ln_2(h))


class Transformer:
    def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
        self.vocab_size = vocab_size
        self.wte = Embedding(vocab_size, dim)
        self.wpe = Embedding(vocab_size, dim)
        self.h = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim, norm_eps)
        self.lm_head = Linear(dim, vocab_size, bias=False)
        self.forward_jit = TinyJit(self.forward)

    def forward(self, tokens: Union[Tensor, Variable], start_pos: Variable, temperature: float = 0.0) -> Tensor:
        if not hasattr(self, "allpos"):
            self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
        if isinstance(tokens, Variable):
            seqlen = 1
            tok_emb = self.wte.weight.shrink(((tokens, tokens + 1), None))
        else:
            seqlen = tokens.shape[1]
            tok_emb = self.wte(tokens)

        pos_emb = self.wpe(self.allpod.shrink((None, (start_pos, start_pos + seqlen))))
        h = tok_emb + pos_emb

        mask = (
            Tensor.full((1, 1, seqlen, start_pos.val + seqlen), float("-inf"), dtype=h.dtype) .triu(start_pos.val + 1)
            if seqlen > 1 else None
        )

        for hi in self.h:
            hi(h, start_pos, mask)

        logits = self.lm_head(self.ln_f(h))

        if logits.shape[1] == 0:
            # speical case for empty prompt
            logits = Tensor.ones((logits.shape[0], self.vocab_size), dtype=logits.dtype, device=logits.device)
        else:
            logits = logits[:, -1, :]

        if temperature < 1e-6:
            ret = logits.argmax(-1)
        else:
            ret = (logits / temperature).softmax().multinomial()
        return ret.flatten().realize()

    def __call__(self, tokens: Tensor, start_pos: Variable, temperature: float = 0.0) -> Tensor:
        forward = (self.forward_jit if JIT and (isinstance(tokens, Variable) or tokens.shape[1] == 1) else self.foward)
        return forward(tokens, start_pos, temperature)


class GPT2:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_length: int, temperature: float, batch_size: int = 1):
        prompt_tokens = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        toks = [prompt_tokens[:] for _ in range(batch_size)]
        start_pos = 0
        for _ in trange(max_length):
            if batch_size == 1 and len(toks[0][start_pos:]) == 1:
                tokens = Variable("tokens", 0, VOCAB_SIZE).bind(toks[0][start_pos])
            else:
                tokens = Tensor([x[start_pos:] for x in toks])
            tok = (
                self.model(tokens, Variable("start_pos", 1 if start_pos else 0), MAX_CONTEXT)
                    .bind(start_pos, temperature)
                    .numpy()
                    .tolist()
            )
            start_pos = len(toks[0])
            for i, t in enumerate(tok):
                toks[i].append(t)
            return [self.tokenizer.decode(x) for x in toks]
