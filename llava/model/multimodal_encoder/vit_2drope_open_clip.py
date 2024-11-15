import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import os
import pdb
from transformers import PretrainedConfig,PreTrainedModel,set_seed,AutoModel
from typing import List
import sys
from itertools import repeat
import collections.abc

from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(module, module_match={}, name=''):
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


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
    #tensor:[10, 577, 16, 64]
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




class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor = None, rotary_pos_emb: torch.Tensor = None
) -> torch.Tensor:
        #NLD -> LND
        seq_length = hidden_states.shape[0]
        batch_size = hidden_states.shape[1]
      

        # Compute Q, K, V matrices
        # Shape: [seq_length, batch_size, dim * 3]
        qkv = self.in_proj(hidden_states)
        # # [seq_length, batch_size, 3, num_heads, head_dim]
        qkv = qkv.view(seq_length, batch_size, 3, self.num_heads, -1)
        # # [3, batch_size, seq_length, num_heads, head_dim]
        qkv = qkv.permute(2, 1, 0, 3, 4)
        # # Each of shape: [batch_size, seq_length, num_heads, head_dim]
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
        attn_output = attn_output.view(seq_length, batch_size, -1)  # [seq_length, batch_size, dim]
        attn_output = self.out_proj(attn_output)
        #attn_output = attn_output.view(batch_size,seq_length, -1) # [seq_length, batch_size, num_heads, head_dim]
        
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
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            attn_type = 'vision',
    ):
        super().__init__()
        self.attn_type = attn_type
        self.ln_1 = LayerNorm(d_model)
        if attn_type == 'vision':
            self.attn = VisionSdpaAttention(d_model, n_head)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, rotary_pos_emb: Optional[torch.Tensor] = None):
        if self.attn_type == 'vision':
            assert rotary_pos_emb is not None
            return self.attn(x, rotary_pos_emb=rotary_pos_emb)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                rotary_pos_emb: Optional[torch.Tensor] = None):

        if rotary_pos_emb is not None:
            x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask, rotary_pos_emb=rotary_pos_emb))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask))
            x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU, attn_type='text'):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = True
        self.attn_type = attn_type

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, attn_type=attn_type)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, rotary_pos_emb: Optional[torch.Tensor] = None,output_hidden_states: bool = False):
        encoder_states = () if output_hidden_states else None
        if self.attn_type == 'vision':
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(r, x, attn_mask, rotary_pos_emb)
                else:
                    x = r(x, attn_mask=attn_mask, rotary_pos_emb=rotary_pos_emb)
                if output_hidden_states:
                    encoder_states = encoder_states + (x.permute(1, 0, 2),)
            if output_hidden_states:
                return encoder_states
            else:
                return x
        else:
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(r, x, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x


class VisualTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.spatial_merge_size = 1

        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer, attn_type='vision')

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.vision_rotary_embedding = VisionRotaryEmbedding(width // heads // 2)
        
        self.class_pos_emb = nn.Parameter(torch.randn(1, width // heads // 2))


    def rot_pos_emb(self, grid_thw):
        pos_ids = []    
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        #24,grid_thw[10,24,24]
        max_grid_size = grid_thw[:, 1:].max()
      
        rotary_pos_emb_full = self.vision_rotary_embedding(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        #[10*24*24, 32/dim])
        return rotary_pos_emb


    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=False):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor, twh = None):
        if twh is None:
            twh = (1, x.size(3) // self.patch_size[0], x.size(2) // self.patch_size[1])
        
        rotary_pos_emb = self.rot_pos_emb(torch.tensor([twh], device=x.device))
        rotary_pos_emb = torch.cat([self.class_pos_emb.expand(twh[0],-1), rotary_pos_emb], dim=0)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, rotary_pos_emb=rotary_pos_emb, output_hidden_states=True)
        return x



def ViT_g_32_1024(input_resolution=224):
    vision_transformer = VisualTransformer(
        image_size=input_resolution, patch_size=32, width=1408,
        layers=40, heads=16, mlp_ratio=4, output_dim=1024)
    return vision_transformer



"""
MLCDAnyResConfig/MLCDAnyRes
"""
class MLCDAnyResConfig(PretrainedConfig):
    model_type = "MLCDAnyRes"
    def __init__(self,hidden_size=1408,patch_size=32,image_size=512,stem_type="",**kwargs):
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")
        self.stem_type = stem_type
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size
        super().__init__(**kwargs)

class MLCDAnyRes(PreTrainedModel):
    config_class = MLCDAnyResConfig
    base_model_prefix = "MLCDAnyRes"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.vision_model = ViT_g_32_1024(input_resolution=512)

    def load_weight(self,weights_path):
        state_dict = torch.load(weights_path, "cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        if "positional_embedding" in state_dict.keys():
            state_dict.pop('positional_embedding')
        self.vision_model.load_state_dict(state_dict)
        

    def forward(self,imgs, twh =None) -> torch.Tensor:
        return self.vision_model(imgs,twh)

 

    

    	

