import collections.abc

from collections import OrderedDict

from itertools import repeat
from typing import Callable


import torch
import torch.nn.functional as F
from torch import nn
from torch import nn as nn




# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


# if "in_proj_weight" in k:
#     convert[k.replace("in_proj_weight", "qkv.weight")] = v
# elif "in_proj_bias" in k:
#     convert[k.replace("in_proj_bias", "qkv.bias")] = v
# elif "out_proj.weight" in k:
#     convert[k.replace("out_proj.weight", "proj.weight")] = v
# elif "out_proj.bias" in k:
#     convert[k.replace("out_proj.bias", "proj.bias")] = v

class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor = None, rotary_pos_emb: torch.Tensor = None
) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        batch_size = hidden_states.shape[1]

        # Compute Q, K, V matrices
        # Shape: [seq_length, batch_size, dim * 3]
        qkv = self.in_proj(hidden_states)
        # [seq_length, batch_size, 3, num_heads, head_dim]
        qkv = qkv.view(seq_length, batch_size, 3, self.num_heads, -1)
        # [3, batch_size, seq_length, num_heads, head_dim]
        qkv = qkv.permute(2, 1, 0, 3, 4)
        # Each of shape: [batch_size, seq_length, num_heads, head_dim]
        q, k, v = qkv.unbind(0)

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
        attention_mask = None
        q = q.permute(0, 2, 1, 3).contiguous() 
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        # q (batch_size, num_heads, seq_length, head_dim)
        # k (batch_size, num_heads, seq_length, head_dim)
        # v (batch_size, num_heads, seq_length, head_dim)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  # [seq_length, batch_size, num_heads, head_dim]
        attn_output = attn_output.view(seq_length, batch_size, -1)  # [seq_length, batch_size, embedding_dim]
        attn_output = self.out_proj(attn_output)
        return attn_output


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, d_model, n_head, mlp_ratio, act_layer: Callable = nn.GELU,):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = VisionSdpaAttention(d_model, n_head)
        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(self, x, rotary_pos_emb):
        return self.attn(x, rotary_pos_emb=rotary_pos_emb)


    def forward(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor):
        x = x + self.attention(self.ln_1(x), rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, mlp_ratio, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x, rotary_pos_emb):
        for r in self.resblocks:
            x = r(x, rotary_pos_emb=rotary_pos_emb)
        return x


class VisualTransformer(nn.Module):
    def __init__(self, patch_size, width, layers, heads, mlp_ratio, act_layer: Callable = nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_size = patch_size
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)
        self.ln_post = LayerNorm(width)
        self.vision_rotary_embedding = VisionRotaryEmbedding(width // heads // 2)
        self.class_pos_emb = nn.Parameter(torch.randn(1, width // heads // 2))


    def rot_pos_emb(self, grid_thw):
        pos_ids = []    
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(h, 1, w, 1,)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(h, 1, w, 1,)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.vision_rotary_embedding(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb


    def forward(self, x: torch.Tensor, twh = None):
        if twh is None:
            twh = (1, x.size(3) // self.patch_size, x.size(2) // self.patch_size)
        rotary_pos_emb = self.rot_pos_emb(torch.tensor([twh], device=x.device))
        rotary_pos_emb = torch.cat([self.class_pos_emb, rotary_pos_emb], dim=0)
        # shape = [*, width, grid, grid]
        x = self.conv1(x) 
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # shape = [*, grid ** 2, width]
        x = x.permute(0, 2, 1)
        # shape = [*, grid ** 2 + 1, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, rotary_pos_emb=rotary_pos_emb)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        return x


def ViT_L_14_336px():
    vision_transformer = VisualTransformer(
        patch_size=14, width=1024, layers=24, heads=16, mlp_ratio=4)
    return vision_transformer


def ViT_g_32_512px():
    vision_transformer = VisualTransformer(
        patch_size=32, width=1408, layers=40, heads=16, mlp_ratio=4)
    return vision_transformer


def ViT_g_32_anyres():
    vision_transformer = VisualTransformer(
        patch_size=32, width=1408, layers=40, heads=16, mlp_ratio=4)
    return vision_transformer
