from easydict import EasyDict
import torchsparse
import torch.nn as nn

# public
__all__ = [
    "model_block_paras",
    "linear",
    "TimestepEmbedSequential",
    "timestep_embedding",
    "zero_module",
    "conv_nd",
    "normalization",
    "ResBlock",
    "Downsample",
    "Upsample",
    "AttentionBlock",
    "to_sparse",
    "to_dense",
    "torchsparse",
    "nonlinear",
    "SpatialTransformer",
]

model_block_paras = EasyDict(dict(use_sparse=False, use_bev=False))

from model.ms_ldm.blocks.blk_modules import (
    linear,
    TimestepEmbedSequential,
    zero_module,
    timestep_embedding,
)

# dense
from model.ms_ldm.blocks.blk_modules import conv_nd as conv_nd_dense
from model.ms_ldm.blocks.blk_modules import ResBlock as ResBlockDense
from model.ms_ldm.blocks.blk_modules import Downsample as DownsampleDense
from model.ms_ldm.blocks.blk_modules import Upsample as UpsampleDense
from model.ms_ldm.blocks.blk_modules import normalization as normalization_dense
from model.ms_ldm.blocks.blk_modules import AttentionBlock as AttentionBlockDense

# sparse
from model.utils.torch_sparse_utils import to_sparse, to_dense
from model.ms_ldm.blocks.sparse_blk_modules import sparse_conv3d, ResBlock3DSparse
from model.ms_ldm.blocks.sparse_blk_modules import Downsample as DownsampleSparse
from model.ms_ldm.blocks.sparse_blk_modules import Upsample as UpsampleSparse
from model.ms_ldm.blocks.sparse_blk_modules import normalization as normalization_sparse
from model.ms_ldm.blocks.sparse_blk_modules import AttentionBlock as AttentionBlockSparse
from model.ms_ldm.blocks.sparse_blk_modules import SiLU as SparseSiLU
from model.ms_ldm.spatial_transformer.spatial_transformer_3d_sparse import (
    SpatialTransformer as SpatialTransformerSparse,
)
from model.ms_ldm.spatial_transformer.spatial_transformer_3d import (
    SpatialTransformer as SpatialTransformerDense,
)
from model.ms_ldm.spatial_transformer.spatial_transformer_2d import (
    SpatialTransformer as SpatialTransformer2D,
)
from model.ms_ldm.spatial_transformer.spatial_transformer_bev_sparse import (
    SpatialTransformer as SpatialTransformerSparseBEV,
)


def nonlinear():
    if model_block_paras.use_sparse:
        return SparseSiLU()
    else:
        return nn.SiLU()


def conv_nd(dims, *args, **kwargs):
    if model_block_paras.use_sparse:
        if dims != 3:
            raise NotImplementedError
        else:
            return sparse_conv3d(*args, **kwargs)
    else:
        return conv_nd_dense(dims, *args, **kwargs)


def normalization(*args, **kwargs):
    if model_block_paras.use_sparse:
        return normalization_sparse(*args, **kwargs)
    else:
        return normalization_dense(*args, **kwargs)


def ResBlock(*args, **kwargs):
    if model_block_paras.use_sparse:
        if kwargs["dims"] != 3:
            raise NotImplementedError
        return ResBlock3DSparse(*args, **kwargs)
    else:
        return ResBlockDense(*args, **kwargs)


def Downsample(*args, **kwargs):
    if model_block_paras.use_sparse:
        if kwargs["dims"] != 3:
            raise NotImplementedError
        return DownsampleSparse(*args, **kwargs)
    else:
        return DownsampleDense(*args, **kwargs)


def Upsample(*args, **kwargs):
    if model_block_paras.use_sparse:
        if kwargs["dims"] != 3:
            raise NotImplementedError
        return UpsampleSparse(*args, **kwargs)
    else:
        return UpsampleDense(*args, **kwargs)


def AttentionBlock(*args, **kwargs):
    if model_block_paras.use_sparse:
        return AttentionBlockSparse(*args, **kwargs)
    else:
        return AttentionBlockDense(*args, **kwargs)


def SpatialTransformer(*args, **kwargs):
    if model_block_paras.use_sparse:
        if model_block_paras.use_bev:
            return SpatialTransformerSparseBEV(*args, **kwargs)
        else:
            return SpatialTransformerSparse(*args, **kwargs)
    else:
        if kwargs["dims"] == 2:
            return SpatialTransformer2D(*args, **kwargs)
        elif kwargs["dims"] == 3:
            return SpatialTransformerDense(*args, **kwargs)
        else:
            raise NotImplementedError
