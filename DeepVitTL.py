# -*- coding: utf-8 -*- #
"""
@Project    ：NIR-Mathematical-Modeling-Tool 
@File       ：VitNet.py 
@Author     ：ZAY
@Time       ：2023/4/9 13:23
@Annotation : "vision transformer 模型 "
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    """层规范化"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """前馈神经网络（MLP）"""
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class ReAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
        # 待学习的转换矩阵参数

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )
        # 新加入的Norm层

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # Calculate attention weight

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)
        # 计算新的attention weight

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ReAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class DeepViTTL(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) # end - start x 1
        patch_height, patch_width = pair(patch_size) # psize x 1
        # assert expression [, arguments] 等价于 if not expression: raise AssertionError(arguments) 条件为 false 的时候触发异常
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) # patches数量
        patch_dim = channels * patch_height * patch_width # 一个patch的维度 1 x patch_height x 1
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 处理原始图像
        # Rearrange：按照字符串的含义对目标进行重新排列的操作
        # img = torch.randn(1, 3, 256, 256)  '1 3 (256 32) (256 32) -> 1 (256 256) (32 32 3)'
        # 分成8*8个patch，并平铺成64个patch，每一个patch的大小为32*32
        # 然后经过一个全连接层(3072, 1024)输出处理后的图像(三维：32*32*3=3072)
        # 光谱 image_size = (2040, 1) psize = (60, 1) 'b 1 (2040 60) (1 1) -> b (2040 1) (60 1 1)' 分成 34x1个patch，每一个patch的大小为60x1，然后经过一个全连接层(60, dim)输出处理后的光谱
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 位置嵌入（patches+cls的位置信息）
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        # 不区分参数的占位符标识运算符。
        # identity模块不改变输入，直接return input
        # 一种编码技巧吧，比如我们要加深网络，有些层是不改变输入数据的维度的，这时就可以使用此函数
        # 这个网络层的设计是仅用于占位的，即不干活，只是有这么一个层，放到残差网络里就是在跳过连接的地方用这个层，显得没有那么空虚！
        self.to_latent = nn.Identity()
        # 定义MLP分类头的模型结构
        # 首先经过LN，然后经过一个全连接层(dim, 1)，输出分类结果
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        # 平铺后，共有(image_height // patch_height) * (image_width // patch_width)个patch，其大小为patch_height*patch_width
        # print('平铺 x.size:', x.size()) # [32, 103, 1024] [batch_size, token_sum, dim]
        # print('尺寸 x.shape:', x.shape) # [32, 103, 1024]
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # print('cls_tokens.size:',cls_tokens.size()) # [32, 1, 1024]
        x = torch.cat((cls_tokens, x), dim=1) # 沿着dim=1方向对cls_token和x进行拼接
        x += self.pos_embedding[:, :(n + 1)] # 拼接后的x与cls嵌入位置信息
        x = self.dropout(x)
        # print('输入transformer前 x.size())', x.size()) # [32, 104, 1024]
        x = self.transformer(x)
        # print('输入transformer后 x.size())', x.size()) # [32, 104, 1024]
        # 如果pool == 'mean'，返回dim = 1方向上的元素平均值
        # 否则，直接返回dim=0方向上的第一行的所有元素，即class tokens
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.mlp(x)
        return x

# v = ViT(
#     num_classes = 10,
#     image_size = (2000, 1),  # image size is a tuple of (height, width)
#     patch_size = (20, 1),    # patch size is a tuple of (height, width)
#     dim = 1024,
#     depth = 5,
#     heads = 20,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )



# from visrecord import Recorder
# v = Recorder(v)
#
#
# # forward pass now returns predictions and the attention maps
# img = torch.randn(4, 1, 2000,1)
# preds, attns = v(img)
#
# # there is one extra patch due to the CLS token
#
# attns(4, 5, 20, 2000, 1)     # (1, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)
#
# preds = v(img)
# print('preds shape:{}'.format(preds.shape))
# print('preds:{}'.format(preds))
