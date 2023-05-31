# -*- coding: utf-8 -*- #
"""
@Project    ：NIR-Mathematical-Modeling-Tool 
@File       ：SpectFormerTL.py
@Author     ：ZAY
@Time       ：2023/5/17 10:19
@Annotation : " "
"""

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential

_logger = logging.getLogger(__name__)


def _cfg(url = '', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h = 14, w = 8):
        super().__init__()
        # h = img_size // patch_size
        # w = h // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype = torch.float32) * 0.02)
        self.w = w
        self.h = h
        # print("h", h, "w", w)
    def forward(self, x, spatial_size = None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        # print("view",x.shape)
        # print("a",a,"b",b)
        x = x.view(B, a, b, C)

        x = x.to(torch.float32)
        # print("rfft2",x.shape)
        # 二维离散傅里叶变换
        # dim(元组[int],可选的) - 要转换的维度。默认值：最后两个维度。
        # norm(str,可选的) - 标准化模式。
        # 对于前向变换(rfft2())
        # "forward" - 通过 1/n 标准化
        # "backward" - 没有标准化
        # "ortho" - 通过1/sqrt(n) 归一化(使真正的 FFT 正交化)
        x = torch.fft.rfft2(x, dim = (1, 2), norm = 'ortho')
        # print("rfft2", x.shape)
        # torch.view_as_complex
        # 把一个tensor转为复数形式，要求这个tensor的最后一个维度形状为2。
        # torch.view_as_complex(torch.Tensor([[1, 2], [3, 4], [5, 6]]))
        # # tensor([1.+2.j, 3.+4.j, 5.+6.j])
        weight = torch.view_as_complex(nn.Parameter(self.complex_weight))
        # print("weight",weight.shape)
        x = x * weight
        # 二维离散傅里叶反向变换
        # s(元组[int], 可选的) - 转换维度中的信号大小。
        x = torch.fft.irfft2(x, s = (a, b), dim = (1, 2), norm = 'ortho')
        # print("irfft2", x.shape)

        x = x.reshape(B, N, C)

        return x

# Spectral Block
class Block(nn.Module):

    def __init__(self, dim, mlp_ratio = 4., drop = 0., drop_path = 0., act_layer = nn.GELU, norm_layer = nn.LayerNorm,
                 h = 14, w = 8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h = h, w = w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

# Attention Block
class Block_attention(nn.Module):

    def __init__(self, dim, mlp_ratio = 4., drop = 0., drop_path = 0., act_layer = nn.GELU, norm_layer = nn.LayerNorm,
                 h = 14, w = 8):
        super().__init__()
        num_heads = 6  # 4 for tiny, 6 for small and 12 for base
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)
        self.attn = Attention(dim, num_heads = num_heads, qkv_bias = True, qk_scale = False, attn_drop = drop,
                              proj_drop = drop)

    def forward(self, x):
        # x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, embed_dim = 768):
        super().__init__()
        img_size = (img_size, 1)
        patch_size = (patch_size, 1)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # print("PatchE",x.shape)
        x = self.proj(x)
        # print("PatchE",x.shape)
        x = x.flatten(2).transpose(1, 2) # B C H W -> B C N -> B N C
        return x


class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size = 56, dim_in = 64, dim_out = 128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size = 2, stride = 2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x


class SpectFormerTL(nn.Module):

    def __init__(self, img_size = 1936, patch_size = 16, in_chans = 1, num_classes = 1, embed_dim = 768, depth = 12,
                 mlp_ratio = 4., representation_size = None, uniform_drop = False,
                 drop_rate = 0., drop_path_rate = 0., norm_layer = None,
                 dropcls = 0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps = 1e-6)

        self.patch_embed = PatchEmbed(
            img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p = drop_rate)

        # h = img_size // patch_size
        # w = h // 2 + 1
        h = int(math.sqrt((img_size - patch_size) / patch_size + 1))
        if h % 2 == 0:
            w = h / 2
        else:
            w = h // 2 + 1



        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        alpha = 4
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i < alpha:
                layer = Block(dim = embed_dim, mlp_ratio = mlp_ratio, drop = drop_rate, drop_path = dpr[i],
                              norm_layer = norm_layer, h = h, w = w)
                self.blocks.append(layer)
            else:
                layer = Block_attention(dim = embed_dim, mlp_ratio = mlp_ratio, drop = drop_rate, drop_path = dpr[i],
                                        norm_layer = norm_layer, h = h, w = w)
                self.blocks.append(layer)

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.head =  nn.Sequential(
            nn.Linear(self.num_features, 96),
            nn.Linear(96, num_classes)
        )

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p = dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std = .02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool = ''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        # print("mean",x.shape) # torch.Size([16, 121, 768])
        x = self.norm(x).mean(1)
        # print("mean", x.shape) # torch.Size([16, 768])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size = (gs_new, gs_new), mode = 'bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim = 1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict
