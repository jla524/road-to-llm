from tinygrad.tensor import Tensor
from road_to_llm.models.transformer import TransformerBlock

class ViT:
  def __init__(self, layers=12, embed_dim=192, num_heads=3):
    self.embedding = (Tensor.uniform(embed_dim, 3, 16, 16), Tensor.zeros(embed_dim))
    self.embed_dim = embed_dim
    self.cls = Tensor.ones(1, 1, embed_dim)
    self.pos_embedding = Tensor.ones(1, 197, embed_dim)
    self.tbs = [
      TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*4,
        prenorm=True, act=lambda x: x.gelu())
      for i in range(layers)]
    self.encoder_norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))
    self.head = (Tensor.uniform(embed_dim, 1000), Tensor.zeros(1000))

  def patch_embed(self, x):
    x = x.conv2d(*self.embedding, stride=16)
    x = x.reshape(shape=(x.shape[0], x.shape[1], -1)).permute(order=(0,2,1))
    return x

  def forward(self, x):
    ce = self.cls.add(Tensor.zeros(x.shape[0],1,1))
    pe = self.patch_embed(x)
    x = ce.cat(pe, dim=1)
    x = x.add(self.pos_embedding).sequential(self.tbs)
    x = x.layernorm().linear(*self.encoder_norm)
    return x[:, 0].linear(*self.head)
