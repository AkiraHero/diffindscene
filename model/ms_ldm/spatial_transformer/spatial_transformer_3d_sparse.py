from inspect import isfunction
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

import torchsparse.nn as spnn
from torchsparse.tensor import SparseTensor

from model.utils.torch_sparse_utils import GroupNorm
from model.ms_ldm.blocks.model_utils import checkpoint
from model.ms_ldm.blocks.sparse_blk_modules import LayerNorm, FeedForward

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# class LayerNorm(nn.LayerNorm):
#     def forward(self, input):
#         coords, feats, stride = input.coords, input.feats, input.stride

#         batch_size = torch.max(coords[:, -1]).item() + 1
#         num_channels = feats.shape[1]

#         nfeats = torch.zeros_like(feats)
#         for k in range(batch_size):
#             indices = coords[:, -1] == k
#             bfeats = feats[indices]
#             # bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
#             bfeats = super().forward(bfeats)
#             # bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
#             nfeats[indices] = bfeats

#         output = SparseTensor(coords=coords, feats=nfeats, stride=stride)
#         try:
#             output.cmaps = input.cmaps
#             output.kmaps = input.kmaps
#         except:
#             output._caches.cmaps = input._caches.cmaps
#             output._caches.kmaps = input._caches.kmaps
#         return output


# class FeedForward(nn.Module):
#     def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
#         super().__init__()
#         inner_dim = int(dim * mult)
#         dim_out = default(dim_out, dim)
#         project_in = (
#             nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
#             if not glu
#             else GEGLU(dim, inner_dim)
#         )

#         self.net = nn.Sequential(
#             project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
#         )

#     def forward(self, x):
#         nfeats = torch.zeros_like(x.F)
#         coords = x.C
#         # cmaps = x.cmaps
#         try:
#             cmaps = x.cmaps
#             kmaps = x.kmaps
#         except:
#             cmaps = x._caches.cmaps
#             kmaps = x._caches.kmaps
#         stride = x.stride
#         batch_inx = x.C[:, -1].unique()

#         for i in batch_inx:
#             indices = x.C[:, -1] == i
#             f = x.F[indices]
#             out = self.net(f)
#             nfeats[indices] = out.squeeze(0)

#         output = SparseTensor(coords=coords, feats=nfeats, stride=stride)
#         try:
#             output.cmaps = cmaps
#             output.kmaps = kmaps
#         except:
#             output._caches.cmaps = cmaps
#             output._caches.kmaps = kmaps
#         return output


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        nfeats = torch.zeros_like(x.F)
        coords = x.C
        # cmaps = x.cmaps
        try:
            cmaps = x.cmaps
            kmaps = x.kmaps
        except:
            cmaps = x._caches.cmaps
            kmaps = x._caches.kmaps
        stride = x.stride

        h = self.heads

        # loop over batch
        batch_inx = x.C[:, -1].unique()
        for i in batch_inx:
            context_i = context[i : i + 1] if context is not None else None
            indices = x.C[:, -1] == i
            f = x.F[indices]
            f = f.unsqueeze(0)

            q = self.to_q(f)
            context_i = default(context_i, f)
            k = self.to_k(context_i)
            v = self.to_v(context_i)

            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
            )

            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
            out = self.to_out(out)

            nfeats[indices] = out.squeeze(0)

        output = SparseTensor(coords=coords, feats=nfeats, stride=stride)
        try:
            output.cmaps = cmaps
            output.kmaps = kmaps
        except:
            output._caches.cmaps = cmaps
            output._caches.kmaps = kmaps
        return output


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        positional_encoding=False,
        spatial_dim=None,
        **args
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = spnn.Conv3d(in_channels, inner_dim, kernel_size=1, stride=1)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(
            spnn.Conv3d(inner_dim, in_channels, kernel_size=1, stride=1)
        )
        self.use_position_encoding = positional_encoding
        if positional_encoding:
            self.positional_embedding = nn.Parameter(
                torch.randn(*spatial_dim, inner_dim) / inner_dim**0.5
            )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # b, c, h, w, l = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        # x = rearrange(x, 'b c h w l -> b (h w l) c')
        if self.use_position_encoding:
            pos_emb = self.positional_embedding[
                x.C[:, 1].to(torch.long),
                x.C[:, 2].to(torch.long),
                x.C[:, 3].to(torch.long),
                :,
            ]
            x.F = x.F + pos_emb
        for block in self.transformer_blocks:
            x = block(x, context=context)
        # x = rearrange(x, 'b (h w l) c -> b c h w l', h=h, w=w, l=l)
        x = self.proj_out(x)
        return x + x_in
