import math
import torch
import numpy as np
from inspect import isfunction

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply

from model.ms_ldm.blocks.blk_modules import TimestepBlock
from model.ms_ldm.blocks.model_utils import linear
from model.utils.torch_sparse_utils import (
    GroupNorm,
    inherit_sparse_tensor,
    get_batch_dim,
)


class SiLU(nn.SiLU):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class SpDropout(nn.Dropout):
    def forward(self, input):
        return fapply(input, super().forward)


class SpIdentity(nn.Identity):
    def forward(self, input):
        return fapply(input, super().forward)


def normalization(channels):
    return GroupNorm(32, channels)


def sparse_conv3d(*args, **kwargs):
    if "padding" in kwargs:
        kwargs.pop("padding")
    return spnn.Conv3d(*args, **kwargs)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=3, out_channels=None, **kwargs):
        super().__init__()
        assert dims == 3
        assert use_conv
        self.channels = channels
        self.out_channels = out_channels or channels
        self.layers = nn.Sequential(
            spnn.Conv3d(self.channels, self.out_channels, 3, stride=2, transposed=True),
            # normalization(self.out_channels),
            # SiLU(True),
        )

    def forward(self, x):
        return self.layers(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    # todo: note the downsampling when dim =3

    def __init__(self, channels, use_conv, dims=3, out_channels=None, **kwargs):
        super().__init__()
        assert dims == 3
        assert use_conv
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2

        self.layers = nn.Sequential(
            spnn.Conv3d(
                in_channels=self.channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=stride,
                dilation=1,
            ),
            # normalization(self.out_channels),
            # SiLU(True),
        )

    def forward(self, x):
        return self.layers(x)


class ResBlock3DSparse(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        up=False,
        down=False,
        enable_emb=True,
        **kwargs,
    ):
        super().__init__()

        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.enable_emb = enable_emb

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            spnn.Conv3d(
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
            ),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = SpIdentity()
        if enable_emb:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    (
                        2 * self.out_channels
                        if use_scale_shift_norm
                        else self.out_channels
                    ),
                ),
            )
        self.out_layers = nn.Sequential(
            # SpDropout(p=dropout),
            # zero_module(
            # maybe for paras init
            normalization(self.out_channels),  # 32 group?
            SiLU(),
            SpDropout(p=dropout),
            spnn.Conv3d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
            ),
            # ),
        )

        if self.out_channels == channels:
            self.skip_connection = SpIdentity()
        elif use_conv:
            self.skip_connection = spnn.Conv3d(
                channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
            )
        else:
            self.skip_connection = spnn.Conv3d(channels, self.out_channels, 1)
        # else:
        #     self.skip_connection = spnn.Conv3d(
        #         channels,
        #         self.out_channels,
        #         kernel_size=1,
        #         stride=1,
        #         dilation=1,
        #     )

    def forward(self, x, emb=None):
        if self.enable_emb:
            return self.forward_emb(x, emb)
        else:
            return self.forward_no_emb(x)

    def forward_emb(self, x, emb):
        batch_dim = get_batch_dim()
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=2)
            h = out_norm(h)
            nfeats = torch.zeros_like(h.F)
            coords = h.C
            batch_inx = h.C[:, batch_dim].unique()
            for i in batch_inx:
                scale_b = scale[i]
                shift_b = shift[i]
                indices = h.C[:, batch_dim] == i
                nfeats[indices] = h.F[indices] * (1 + scale_b) + shift_b
            output = inherit_sparse_tensor(h, coords, nfeats)
            output = out_rest(output)
        else:
            nfeats = torch.zeros_like(h.F)
            coords = h.C
            batch_inx = h.C[:, batch_dim].unique()
            for i in batch_inx:
                emb_out_b = emb_out[i]
                indices = h.C[:, batch_dim] == i
                nfeats[indices] = h.F[indices] + emb_out_b
            output = inherit_sparse_tensor(h, coords, nfeats)
            output = self.out_layers(output)
        return self.skip_connection(x) + output

    def forward_no_emb(self, x):
        h = self.in_layers(x)
        output = self.out_layers(h)
        return self.skip_connection(x) + output


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        # self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        batch_dim = get_batch_dim()
        nfeats = torch.zeros_like(x.F)
        coords = x.C

        # norm
        x = self.norm(x)
        batch_inx = x.C[:, batch_dim].unique()
        for i in batch_inx:
            # get split features
            indices = x.C[:, batch_dim] == i
            f = x.F[indices]
            f = f.unsqueeze(0).permute([0, 2, 1])
            qkv = self.qkv(f)
            h = self.attention(qkv)
            h = self.proj_out(h)
            out_f = f + h
            nfeats[indices] = out_f.squeeze(0).permute([1, 0])

        output = inherit_sparse_tensor(x, coords, nfeats)
        return output


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
    
'''

for spatial transformers
'''
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class LayerNorm(nn.LayerNorm):
    def forward(self, input):
        coords, feats, stride = input.coords, input.feats, input.stride
        batch_dim = get_batch_dim()
        batch_size = torch.max(coords[:, batch_dim]).item() + 1
        num_channels = feats.shape[1]

        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, batch_dim] == k
            bfeats = feats[indices]
            # bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            # bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        output = SparseTensor(coords=coords, feats=nfeats, stride=stride)
        try:
            output.cmaps = input.cmaps
            output.kmaps = input.kmaps
        except:
            output._caches.cmaps = input._caches.cmaps
            output._caches.kmaps = input._caches.kmaps
        return output
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        batch_dim = get_batch_dim()
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
        batch_inx = x.C[:, batch_dim].unique()

        for i in batch_inx:
            indices = x.C[:, batch_dim] == i
            f = x.F[indices]
            out = self.net(f)
            nfeats[indices] = out.squeeze(0)

        output = SparseTensor(coords=coords, feats=nfeats, stride=stride)
        try:
            output.cmaps = cmaps
            output.kmaps = kmaps
        except:
            output._caches.cmaps = cmaps
            output._caches.kmaps = kmaps
        return output
    

