import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import einsum


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(
        self,
        num_hiddens,
        embedding_dim,
        n_embed,
        straight_through=True,
        kl_weight=1e-8,
        temp_init=1.0,
        use_vqinterface=True,
        remap=None,
        unknown_index="random",
        use_3d=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.use_3d = use_3d
        if use_3d:
            self.proj = nn.Conv3d(num_hiddens, n_embed, 1)
        else:
            self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        if self.use_3d:
            assert len(z.shape) == 5

        logits = self.proj(z)

        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)

        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        if self.use_3d:
            z_q = einsum("b n h w l, n d -> b d h w l", soft_one_hot, self.embed.weight)
        else:
            z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)
        qy = F.softmax(logits, dim=1)
        diff = (
            self.kl_weight
            * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        )

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        if len(shape) == 4:
            b, h, w, c = shape
            assert b * h * w == indices.shape[0]
            indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)
        elif len(shape) == 5:
            b, h, w, l, c = shape
            assert b * h * w * l == indices.shape[0]
            indices = rearrange(indices, "(b h w l) -> b h w l", b=b, h=h, w=w, l=l)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        if len(shape) == 4:
            one_hot = (
                F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
            )
            z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        elif len(shape) == 5:
            one_hot = (
                F.one_hot(indices, num_classes=self.n_embed)
                .permute(0, 4, 1, 2, 3)
                .float()
            )
            z_q = einsum("b n h w l, n d -> b d h w l", one_hot, self.embed.weight)
        return z_q
