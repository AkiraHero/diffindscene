import torch
import torch.nn as nn

import torchsparse
from torchsparse import SparseTensor


def sparse_cat(T1, T2, dim):
    assert T1.stride == T2.stride
    t1, m1 = to_dense(T1)
    t2, m2 = to_dense(T2)
    max_size, _ = torch.stack(
        [torch.tensor(t1.shape[2:]), torch.tensor(t2.shape[2:])]
    ).max(dim=0)

    t1_ = torch.zeros(list(t1.shape[:2]) + list(max_size), device=t1.device)
    t2_ = torch.zeros(list(t2.shape[:2]) + list(max_size), device=t2.device)
    m1_ = torch.zeros(list(m1.shape[:2]) + list(max_size), device=m1.device)
    m2_ = torch.zeros(list(m2.shape[:2]) + list(max_size), device=m2.device)

    t1_[:, :, : t1.shape[2], : t1.shape[3], : t1.shape[4]] += t1
    t2_[:, :, : t2.shape[2], : t2.shape[3], : t2.shape[4]] += t2
    m1_[:, :, : m1.shape[2], : m1.shape[3], : m1.shape[4]] += m1
    m2_[:, :, : m2.shape[2], : m2.shape[3], : m2.shape[4]] += m2

    t = torch.cat([t1_, t2_], dim=dim)
    m1_ = m1_.to(torch.bool)
    m2_ = m2_.to(torch.bool)
    m = m1_ | m2_
    out = to_sparse(t, stride=T1.stride, mask=m, spatial_range=T1.spatial_range)

    return out


def to_dense(input: SparseTensor):
    coords, feats, stride = input.coords, input.feats, input.stride
    coords = coords.t().long()
    if torchsparse.__version__ == "2.1.0":
        pass
    else:
        coords[:3] = (
            coords[:3] / torch.tensor(stride).reshape(-1, 1).long().to(coords)
        ).long()

    coalesce = torch.sparse_coo_tensor(coords, feats).coalesce()
    output = coalesce.to_dense()
    indices = coalesce.indices().t()
    if torchsparse.__version__ == "2.1.0":
        # B * W * H * L * C -> B * C * W * H * L
        output = output.permute(0, 4, 1, 2, 3).contiguous()
        b, w, h, l = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
    else:
        # W * H * L * B * C -> B * C * W * H * L
        output = output.permute(3, 4, 0, 1, 2).contiguous()
        w, h, l, b = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

    B, C, W, H, L = output.shape
    mask = torch.zeros([B, 1, W, H, L], device=feats.device)
    mask[b, :, w, h, l] = 1
    return output, mask


def to_sparse(x, stride=None, mask=None, spatial_range=None):
    if stride is None:
        stride = (1, 1, 1)
    if mask is None:
        C = x.sum(dim=1).nonzero()
    else:
        C = mask.sum(dim=1).nonzero()
    b, w, h, l = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    F = x[b, :, w, h, l]
    if torchsparse.__version__ == "2.1.0":
        C = torch.stack([b, w, h, l]).t().int()
    else:
        C = torch.stack([w * stride[0], h * stride[1], l * stride[2], b]).t().int()

    if torchsparse.__version__ == "2.1.0":
        if spatial_range is None:
            spatial_range = [x.shape[0]] + list(x.shape[2:])
        out = SparseTensor(F, C, stride=stride, spatial_range=spatial_range)
    else:
        out = SparseTensor(F, C, stride=stride)
    return out


def inherit_sparse_tensor(x, coord, feat):
    if torchsparse.__version__ == "2.1.0":
        output = SparseTensor(
            coords=coord, feats=feat, stride=x.stride, spatial_range=x.spatial_range
        )
        cmaps = x._caches.cmaps
        kmaps = x._caches.kmaps
        output._caches.cmaps = cmaps
        output._caches.kmaps = kmaps
    else:
        output = SparseTensor(coords=coord, feats=feat, stride=x.stride)
        cmaps = x.cmaps
        kmaps = x.kmaps
        output.cmaps = cmaps
        output.kmaps = kmaps
    return output


def get_batch_dim():
    if torchsparse.__version__ == "2.1.0":
        return 0
    else:
        return -1


class GroupNorm(nn.GroupNorm):

    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats = input.coords, input.feats

        batch_dim = get_batch_dim()
        batch_size = torch.max(coords[:, batch_dim]).item() + 1

        num_channels = feats.shape[1]

        # PyTorch's GroupNorm function expects the input to be in (N, C, *)
        # format where N is batch size, and C is number of channels. "feats"
        # is not in that format. So, we extract the feats corresponding to
        # each sample, bring it to the format expected by PyTorch's GroupNorm
        # function, and invoke it.
        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, batch_dim] == k
            bfeats = feats[indices]
            bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        output = inherit_sparse_tensor(input, coords, nfeats)

        return output
