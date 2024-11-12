import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize, ToTensor)
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    def __init__(self, input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=768,
                 depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=0.0, using_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.patch_embed = PatchEmbedding(
            input_size, patch_size, in_channels, dim,)
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.num_patches, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, mlp_ratio, dpr[i], self.patch_embed.num_patches, using_checkpoint) for i in range(depth)
            ])
        self.norm = nn.LayerNorm(dim)

        self.feature = nn.Sequential(
            nn.Linear(dim * self.patch_embed.num_patches, dim, False),
            nn.BatchNorm1d(dim, eps=2e-5),
            nn.Linear(dim, embedding_size, False),
            nn.BatchNorm1d(embedding_size, eps=2e-5))

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for func in self.blocks:
            x = func(x)
        x = self.norm(x.float())
        return torch.reshape(x, (B, self.patch_embed.num_patches * self.dim))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.feature(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_hidden)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        with torch.cuda.amp.autocast(True):
            B, L, D = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads,
                                      D // self.num_heads).permute(2, 0, 3, 1, 4)
            # B, L, 3, heads, head_dim ->
            # 3, B, heads, L, head_dim
            q, k, v = qkv[0], qkv[1], qkv[2]
            # q B, heads, L, head_dim

            # q (batch_size, num_heads, seq_length, head_dim)
            # k (batch_size, num_heads, seq_length, head_dim)
            # v (batch_size, num_heads, seq_length, head_dim)
            attn_output = F.scaled_dot_product_attention(q, k, v, None, dropout_p=0.0)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  # [seq_length, batch_size, num_heads, head_dim]
            attn_output = attn_output.view(L, B, -1)  # [seq_length, batch_size, embedding_dim]
            attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_length, embedding_dim]
            x = self.proj(attn_output)

        # with torch.cuda.amp.autocast(False):
        #     q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        #     attn = (q @ k.transpose(-2, -1)) * self.scale
        #     attn = attn.softmax(dim=-1)
        #     x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        # with torch.cuda.amp.autocast(True):
        #     x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, drop_path: float = 0.0, patch_n: int = 32, using_checkpoint=False):
        super().__init__()
        self.using_checkpoint = using_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.extra_gflops = (num_heads * patch_n * (dim // num_heads) * patch_n * 2) / (1000**3)

    def forward_impl(self, x):
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        # if self.using_checkpoint:
        #     return checkpoint(self.forward_impl, x)
        # else:
        #     return self.forward_impl(x)
        return self.forward_impl(x)


class PatchEmbedding(nn.Module):
    def __init__(self, input_size=224, patch_size=32, in_channels: int = 3, dim: int = 768):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        H = input_size[0] // patch_size[0]
        W = input_size[1] // patch_size[1]
        self.num_patches = H * W
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def build_model(name="ViT-L/14@336px"):
    if name == "ViT-B/32":
        model = VisionTransformer(
            input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=512,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=False)
    elif name == "ViT-B/16":
        model = VisionTransformer(
            input_size=224, patch_size=16, in_channels=3, dim=768, embedding_size=768,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=False)
    elif name == "ViT-L/14":
        model = VisionTransformer(
            input_size=224, patch_size=14, in_channels=3, dim=1024, embedding_size=768,
            depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=False)
    elif name == "ViT-L/14@336px":
        model = VisionTransformer(
            input_size=336, patch_size=14, in_channels=3, dim=1024, embedding_size=768,
            depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=False)
    return model


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def load_model_and_transform(name="ViT-L/14@336px"):
    if name == "ViT-B/32":
        return build_model(name), _transform(224)
    elif name == "ViT-B/16":
        return build_model(name), _transform(224)
    elif name == "ViT-L/14":
        return build_model(name), _transform(224)
    elif name == "ViT-L/14@336px":
        return build_model(name), _transform(336)
    else:
        raise
