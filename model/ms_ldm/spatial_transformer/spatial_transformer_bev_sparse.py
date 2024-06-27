from inspect import isfunction
from einops import rearrange
import math

import torch
import torch.nn.functional as F
from torch import nn, einsum

import torchsparse.nn as spnn
from torchsparse.tensor import SparseTensor

from model.utils.torch_sparse_utils import GroupNorm, get_batch_dim, inherit_sparse_tensor
from model.ms_ldm.blocks.model_utils import checkpoint
from model.ms_ldm.blocks.sparse_blk_modules import LayerNorm, FeedForward, default


# feedforward
# class GEGLU(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.proj = nn.Linear(dim_in, dim_out * 2)

#     def forward(self, x):
#         x, gate = self.proj(x).chunk(2, dim=-1)
#         return x * F.gelu(gate)


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d


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
#         batch_dim = get_batch_dim()
#         batch_size = torch.max(coords[:, batch_dim]).item() + 1
#         num_channels = feats.shape[1]

#         nfeats = torch.zeros_like(feats)
#         for k in range(batch_size):
#             indices = coords[:, batch_dim] == k
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
#         batch_dim = get_batch_dim()
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
#         batch_inx = x.C[:, batch_dim].unique()

#         for i in batch_inx:
#             indices = x.C[:, batch_dim] == i
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
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_pe=False,
        spatial_dim=None,
        context_len=None,
        resolution_level=None,
    ):
        super().__init__()
        self.is_self_att = False
        self.context_dim = context_dim
        self.use_pe = use_pe
        if context_dim is None:
            self.is_self_att = True
        else:
            self.self_attd_mapper = nn.Linear(query_dim, context_dim, bias=False)

        self.spatial_dim = None
        self.context_len = None
        self.resolution_level = None
        if self.use_pe:
            assert spatial_dim is not None
            self.spatial_dim = spatial_dim
            self.resolution_level = resolution_level
            if not self.is_self_att:
                assert context_len is not None
                self.context_len = context_len

        inner_dim = dim_head * heads

        self.scale = dim_head**-0.5
        self.heads = heads

        context_dim = default(context_dim, query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        if self.use_pe:
            # bev
            bev_spatial_dim = spatial_dim[:2]
            bev_spatial_dim = [i // self.resolution_level for i in bev_spatial_dim]
            self.pos_embedding = nn.Parameter(
                torch.randn(*bev_spatial_dim, inner_dim) / inner_dim**0.5
            )
            if not self.is_self_att:
                self.context_pos_embedding = nn.Parameter(
                    torch.randn(self.context_len, self.context_dim) / inner_dim**0.5
                )

    def get_pos_emb(self, coord, stride):
        batch_dim = get_batch_dim()
        if batch_dim == -1:
            rescale_coord = torch.floor(
                coord[:, :2] / torch.tensor(stride[:2], device=coord.device)
            )
        elif batch_dim == 0:
            rescale_coord = torch.floor(
                coord[:, 1:3] / torch.tensor(stride[:2], device=coord.device)
            )
        else:
            raise NotImplementedError
        rescale_coord = rescale_coord.to(torch.long)
        # debug
        cmax, _ = rescale_coord.max(dim=0)
        cmin, _ = rescale_coord.min(dim=0)

        assert cmax[0] < self.pos_embedding.shape[0]
        assert cmax[1] < self.pos_embedding.shape[1]
        assert cmin[0] >= 0
        assert cmin[1] >= 0
        return self.pos_embedding[
            rescale_coord[:, 0].to(torch.long), rescale_coord[:, 1].to(torch.long), ...
        ]

    def get_context(self, context_dict):
        assert context_dict["type"] == "sketch_emb"
        emb = context_dict["emb"]
        bs, h, w, c = emb.shape

        if self.use_pe:
            pos_emb = self.context_pos_embedding.view(h, w, -1)
            emb = emb + pos_emb
        return emb.view(bs, -1, c).contiguous()

    def get_context_mask(self, feature, context):
        batch_dim = get_batch_dim()
        if context["type"] == "sketch_emb":
            coord = feature.C.clone()
            stride = feature.stride

            # coord[:, :3] = coord[:, :3] / torch.tensor(feature.stride, device=feature.F.device)
            context_dim = context["emb"].shape[1:3]
            self_dim = self.spatial_dim[0:2]
            ratio = torch.tensor(context_dim, device=feature.F.device) / torch.tensor(
                self_dim, device=feature.F.device
            )
            if batch_dim == -1:
                coord_to_context = coord[:, :2] * ratio
                coord_to_context = torch.cat([coord[:, -1:], coord_to_context], dim=1)
            elif batch_dim == 0:
                coord_to_context = coord[:, 1:3] * ratio
                coord_to_context = torch.cat([coord[:, 0:1], coord_to_context], dim=1)
            else:
                raise NotImplementedError

            ####
            neighbor = [i for i in stride[:2]]
            ####

            
            coordinates = torch.round(coord_to_context).to(torch.long)
            # coordinate index, batch index, x, y
            coordinates = torch.cat(
                [
                    torch.arange(0, coordinates.shape[0])
                    .reshape(-1, 1)
                    .to(coordinates.device),
                    coordinates,
                ],
                dim=1,
            )

            D1, D2 = neighbor
            H, W = context_dim[0], context_dim[1]
            batch_coord_num = coordinates.shape[0]

            kernel_coord = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(-D1, D1 + 1, device=coordinates.device),
                        torch.arange(-D2, D2 + 1, device=coordinates.device),
                    ),
                    dim=0,
                )
                .reshape(2, -1)
                .t()
            )

            mask = torch.zeros(
                batch_coord_num, H, W, dtype=torch.float, device=feature.F.device
            )

            mask_coords = torch.repeat_interleave(coordinates, len(kernel_coord), dim=0)
            kernel_bias = kernel_coord.repeat([coordinates.shape[0], 1])
            mask_coords[:, 2:] = mask_coords[:, 2:] + kernel_bias

            mask_coords[:, 2] = torch.clamp(
                mask_coords[:, 2], min=0, max=context_dim[0] - 1
            )
            mask_coords[:, 3] = torch.clamp(
                mask_coords[:, 3], min=0, max=context_dim[1] - 1
            )

            mask[mask_coords[..., 0], mask_coords[..., 2], mask_coords[..., 3]] = 1
            return mask.view(batch_coord_num, -1).contiguous().to(torch.bool)

        else:
            raise NotImplementedError

    def forward(self, x, context=None):
        batch_dim = get_batch_dim()
        if self.is_self_att:
            assert context is None
        else:
            # when cross-attd are used as self-attd
            assert hasattr(self, "self_attd_mapper")

        nfeats = torch.zeros_like(x.F)
        assert x.stride[0] == x.stride[1] == x.stride[2]

        assert self.resolution_level == x.stride[0]
        coords = x.C
        stride = x.stride

        mask = None
        if context is not None:
            context_data = self.get_context(context)
            mask = self.get_context_mask(feature=x, context=context)

        h = self.heads

        # loop over batch
        batch_inx = x.C[:, batch_dim].unique()
        for i in batch_inx:
            indices = x.C[:, batch_dim] == i
            f = x.F[indices]
            c = x.C[indices]
            f = f.unsqueeze(0)

            if self.use_pe:
                pos_emb = self.get_pos_emb(c, stride)
                f = f + pos_emb

            if context is not None:
                context_i = context_data[i : i + 1]
                if mask is not None:
                    mask_i = mask[indices].unsqueeze(0)
            else:
                if self.is_self_att:
                    context_i = f
                else:
                    context_i = self.self_attd_mapper(f)

            q = self.to_q(f)
            k = self.to_k(context_i)
            v = self.to_v(context_i)

            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
            )

            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

            if exists(mask):
                # mask_i = rearrange(mask_i, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                # mask_i = repeat(mask_i, 'b j -> (b h) () j', h=h)
                mask_i = mask_i.repeat([h, 1, 1])
                sim.masked_fill_(~mask_i, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
            out = self.to_out(out)

            nfeats[indices] = out.squeeze(0)

        output = inherit_sparse_tensor(x, coords, nfeats)
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
        use_pe=False,
        spatial_dim=None,
        resolution_level=None,
        context_len=None,
    ):
        super().__init__()
        self.spatial_dim = None
        self.context_len = context_len
        if use_pe:
            assert spatial_dim is not None
            assert resolution_level is not None
            self.spatial_dim = spatial_dim
            self.resolution_level = resolution_level
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            use_pe=use_pe,
            spatial_dim=self.spatial_dim,
            resolution_level=self.resolution_level,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            use_pe=use_pe,
            spatial_dim=self.spatial_dim,
            context_len=self.context_len,
            resolution_level=self.resolution_level,
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


class ToBEVConvolution(nn.Module):
    """Converts a SparseTensor into a sparse BEV feature map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_kernels: int,
        stride: int = 1,
        dim: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_kernels = n_kernels
        self.stride = stride
        self.dim = dim
        self.kernel = nn.Parameter(torch.zeros(n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if bias else 0
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)

    def extra_repr(self):
        return "in_channels={}, out_channels={}, n_kernels={}, stride={}".format(
            self.in_channels, self.out_channels, self.n_kernels, self.stride
        )

    def forward(self, input: SparseTensor) -> torch.Tensor:
        batch_dim = get_batch_dim()
        coords, feats, stride = input.coords, input.feats, input.stride
        ratio = stride * self.stride

        stride = torch.tensor(stride).unsqueeze(dim=0).to(feats)[:, 2] # z-dim in stride


        kernels = torch.index_select(
            self.kernel, 0, torch.div(coords[:, self.dim].long(), stride).trunc().long()
        )
        feats = (feats.unsqueeze(dim=-1) * kernels).sum(1) + self.bias
        coords = coords.t().long()
        coords[self.dim, :] = 0
        if self.stride > 1:
            if batch_dim == -1:
                coords[:3] /= ratio
                coords[:3] *= ratio
            elif batch_dim == 0:
                coords[1:] /= ratio
                coords[1:] *= ratio

        coalesce = torch.sparse_coo_tensor(coords, feats).coalesce()
        indices = coalesce.indices().t().int()
        values = coalesce.values()
        return SparseTensor(values, indices, ratio)


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
        context_len=None,
        resolution_level=None,
        **args
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_position_encoding = positional_encoding
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.resolution_level = resolution_level

        self.proj_in = spnn.Conv3d(in_channels, inner_dim, kernel_size=1, stride=1)
        self.context_len = None
        if positional_encoding:
            self.context_len = context_len
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_pe=self.use_position_encoding,
                    spatial_dim=spatial_dim,
                    resolution_level=resolution_level,
                    context_len=self.context_len,
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(
            spnn.Conv3d(inner_dim, in_channels, kernel_size=1, stride=1)
        )
        batch_dim = get_batch_dim()
        if batch_dim == 0:
            self.z_dim = 3
        elif batch_dim == -1:
            self.z_dim = 2
        else:
            raise NotImplementedError
        self.bev_adapter = ToBEVConvolution(in_channels, in_channels, 128, dim=self.z_dim) # dim: the dim index of z(up)

    def bev2origin(self, bev_feature, origin_coordinate):
        bev_ = torch.sparse_coo_tensor(bev_feature.C.t().long(), bev_feature.F)
        c = origin_coordinate.clone()
        c[:, self.z_dim] = 0
        c = c.long()
        bev_dense = bev_.to_dense()
        bev_f = bev_dense[c[:, 0], c[:, 1], c[:, 2], c[:, 3]]
        return bev_f

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # b, c, h, w, l = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)

        x_bev = self.bev_adapter(x)

        for block in self.transformer_blocks:
            x_bev = block(x_bev, context=context)
        bev_feature_mapped = self.bev2origin(x_bev, x.C)

        x.F = x.F + bev_feature_mapped
        x = self.proj_out(x)
        return x + x_in
