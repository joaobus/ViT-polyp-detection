import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 patch_size: int = 16, 
                 emb_size: int = 768, 
                 img_size: int = 224,
                 device = torch.device('cpu') ):
        self.patch_size = patch_size
        self.device = device
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x).to(self.device)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b).to(self.device)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1).to(self.device)
        # add position embedding
        x += self.positions
        return x
    


class MSA(nn.Module):
    def __init__(self, 
                 emb_size: int, 
                 num_heads: int = 8, 
                 dropout: float = 0, 
                 return_weights: bool = False):        
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.return_weights = return_weights
        
    def forward(self, x : Tensor, mask: Tensor = None): # type: ignore
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return [out, att] if self.return_weights else out
    


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
    
class FeedForwardBlock(nn.Module):
    def __init__(self, 
                 emb_size: int, 
                 mlp_ratio: int = 4, 
                 drop_p: float = 0.):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, mlp_ratio * emb_size),
            nn.GELU(),                                   # Gaussian Error Linear Unit
            nn.Dropout(drop_p),
            nn.Linear(mlp_ratio * emb_size, emb_size),
            nn.Dropout(drop_p)
        )

    def forward(self, x):
      x = self.ff(x)
      return x
    


class EncoderBlock(nn.Module):
    def __init__(self,
                emb_size: int = 768,
                drop_p: float = 0.,
                forward_drop_p: float = 0.,
                num_heads: int = 8,
                return_attn_weights: bool = False,
                ** kwargs):
      super().__init__()
      self.return_attn_weights = return_attn_weights
      self.msa = nn.Sequential(
          nn.LayerNorm(emb_size),
          MSA(emb_size,
              num_heads,
              dropout=drop_p,
              return_weights=return_attn_weights)
      )
      self.mlp = nn.Sequential(
          nn.LayerNorm(emb_size),
          FeedForwardBlock(emb_size,
                           drop_p=forward_drop_p)
      )
      

    def forward(self, x):
      if self.return_attn_weights:
        x_res, weights = self.msa(x)
        x = x + x_res
      else:
        x = ResidualAdd(self.msa)(x)
      out = ResidualAdd(self.mlp)(x)
      return [out, weights] if self.return_attn_weights else out # type:ignore
    


class VisionTransformer(nn.Module):
  def __init__(self,
               in_channels: int = 3,
               patch_size: int = 16,
               emb_size: int = 768,
               img_size: int = 224,
               drop_p: float = 0.,
               forward_drop_p: float = 0.,
               num_heads: int = 8,
               n_blocks: int = 5,            
               n_classes: int = 1,
               pool: str = 'cls',
               return_attn_weights: bool = False,
               device = torch.device('cpu'),
               ** kwargs):
    
    assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
    assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

    super().__init__()

    self.return_attn_weights = return_attn_weights
    self.pool = pool
    self.patch_embedding = PatchEmbedding(in_channels,
                                          patch_size,
                                          emb_size,
                                          img_size,
                                          device=device)
    self.encoder_blocks = nn.ModuleList(
        [EncoderBlock(emb_size,
                      drop_p,
                      forward_drop_p,
                      num_heads,
                      return_attn_weights) for _ in range(n_blocks)]
    )
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(emb_size),
        nn.Linear(emb_size,n_classes),
#         nn.Softmax(dim=-1)
        nn.Sigmoid()
    )
  

  def forward(self,x):
    x = self.patch_embedding(x)
    for encoder_block in self.encoder_blocks:
        if self.return_attn_weights:
            x, weights = encoder_block(x)
            self.attention_weights = weights
        else:
            x = encoder_block(x)
    x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
    out = self.mlp_head(x)
    return out