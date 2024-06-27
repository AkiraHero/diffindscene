import torch
from torch.nn import init
import torch.nn as nn

from einops import rearrange

from model.model_base import ModelBase

from model.auto_encoder.ms_vqgan.encoder_decoder import (
    Encoder,
    Decoder_occ,
    nonlinearity,
    Upsample,
)

from model.auto_encoder.ms_vqgan.quantize import GumbelQuantize
from model.auto_encoder.ms_vqgan.loss import VQLossWithDiscriminator
from model.auto_encoder.ms_vqgan.lr_scheduler import LambdaWarmUpCosineScheduler


def init_weights(net, init_type="normal", gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # propagate to children
    for m in net.children():
        m.apply(init_func)


class Upsampler4x(nn.Module):
    def __init__(self, in_chn) -> None:
        super().__init__()
        self.up1 = Upsample(in_chn, True)
        self.up2 = Upsample(in_chn, True)

    def forward(self, x):
        x = self.up1(x)
        x = nonlinearity(x)
        x = self.up2(x)
        return x


class Upsampler2x(nn.Module):
    def __init__(self, in_chn) -> None:
        super().__init__()
        self.up1 = Upsample(in_chn, True)

    def forward(self, x):
        x = self.up1(x)
        return x


class MSTSDFPVQGANNew(ModelBase):
    def __init__(self, config):
        super().__init__()
        lossconfig = config.paras.lossconfig
        n_embed = config.paras.n_embed
        embed_dim = config.paras.embed_dim
        self.patch_mode = True
        self.encoder1 = Encoder(**config.paras.ddconfig1)
        self.encoder2 = Encoder(**config.paras.ddconfig2)

        self.decoder1 = Decoder_occ(**config.paras.ddconfig1)
        self.decoder2 = Decoder_occ(**config.paras.ddconfig2)

        lv2_downsample_num = len(config.paras.ddconfig2.ch_mult) - 1
        if lv2_downsample_num == 1:
            self.upsampler = Upsampler2x(config.paras.ddconfig1["z_channels"])
        elif lv2_downsample_num == 2:
            self.upsampler = Upsampler4x(config.paras.ddconfig1["z_channels"])

        self.loss = VQLossWithDiscriminator(**lossconfig.params)

        self.cube_size, self.stride = 4, 4
        if "patch_cube_size" in config.paras:
            self.cube_size, self.stride = (
                config.paras["patch_cube_size"],
                config.paras["patch_cube_size"],
            )

        z_channels = config.paras.ddconfig1["z_channels"]
        self.quantize1 = GumbelQuantize(
            z_channels,
            embed_dim,
            n_embed=n_embed,
            kl_weight=1e-8,
            temp_init=1.0,
            remap=None,
        )

        self.quantize2 = GumbelQuantize(
            z_channels,
            embed_dim,
            n_embed=n_embed,
            kl_weight=1e-8,
            temp_init=1.0,
            remap=None,
        )
        self.temperature_scheduler = LambdaWarmUpCosineScheduler(
            **config.paras.temperature_scheduler_config
        )

        q_dim = embed_dim
        self.quant_conv1 = torch.nn.Conv3d(
            config.paras.ddconfig1["z_channels"] + config.paras.ddconfig2["z_channels"],
            q_dim,
            1,
        )
        self.post_quant_conv1 = torch.nn.Conv3d(
            embed_dim * 2, config.paras.ddconfig1["z_channels"], 1
        )

        self.quant_conv2 = torch.nn.Conv3d(
            config.paras.ddconfig2["z_channels"], q_dim, 1
        )
        self.post_quant_conv2 = torch.nn.Conv3d(
            embed_dim, config.paras.ddconfig2["z_channels"], 1
        )

        if self.training:
            self.init_w()

    def init_w(self):
        init_weights(self.encoder1, "normal", 0.02)
        init_weights(self.encoder2, "normal", 0.02)
        init_weights(self.decoder1, "normal", 0.02)
        init_weights(self.decoder2, "normal", 0.02)
        init_weights(self.upsampler, "normal", 0.02)

        init_weights(self.quant_conv1, "normal", 0.02)
        init_weights(self.post_quant_conv1, "normal", 0.02)
        init_weights(self.quant_conv2, "normal", 0.02)
        init_weights(self.post_quant_conv2, "normal", 0.02)

    @staticmethod
    # def unfold_to_cubes(self, x, cube_size=8, stride=8):
    def unfold_to_cubes(x, cube_size=8, stride=8):
        """
        assume x.shape: b, c, d, h, w
        return: x_cubes: (b cubes)
        """
        x_cubes = (
            x.unfold(2, cube_size, stride)
            .unfold(3, cube_size, stride)
            .unfold(4, cube_size, stride)
        )
        x_cubes = rearrange(x_cubes, "b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w")
        x_cubes = rearrange(x_cubes, "b c p d h w -> (b p) c d h w")
        return x_cubes

    @staticmethod
    # def fold_to_voxels(self, x_cubes, batch_size, ncubes_per_dim):
    def fold_to_voxels(
        x_cubes, batch_size, ncubes_per_dim, ncubes_per_dim2, ncubes_per_dim3
    ):
        x = rearrange(x_cubes, "(b p) c d h w -> b p c d h w", b=batch_size)
        x = rearrange(
            x,
            "b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)",
            p1=ncubes_per_dim,
            p2=ncubes_per_dim2,
            p3=ncubes_per_dim3,
        )
        return x

    def temperature_scheduling(self, global_step):
        self.quantize1.temperature = self.temperature_scheduler(global_step)
        self.quantize2.temperature = self.temperature_scheduler(global_step)

    def decode(self, quant_t, quant_b, occlv1=None, sparse_decode=True):
        upsample_t = self.upsampler(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        if sparse_decode:
            chn = quant.shape[1]
            occ = torch.sigmoid(occlv1)
            occ_mask = occ > 0.5
            occ[occ_mask] = 1
            occ[~occ_mask] = 0
            occ = occ.to(torch.bool)
            occ = occ.repeat([1, chn, 1, 1, 1])
            quant[~occ] *= 0
        quant = self.post_quant_conv1(quant)
        dec, occ_l2 = self.decoder1(quant)
        return dec, occ_l2

    def patch_encode(self, input_data, before_quant=False):
        cur_bs = input_data.shape[0]
        ncubes_per_dim = [i // self.cube_size for i in input_data.shape[2:]]
        x_cubes1 = self.unfold_to_cubes(input_data, self.cube_size, self.stride)
        h1 = self.encoder1(x_cubes1)
        h1_voxel = self.fold_to_voxels(h1, cur_bs, *ncubes_per_dim)
        cur_bs = h1_voxel.shape[0]
        ncubes_per_dim = [i // self.cube_size for i in h1_voxel.shape[2:]]

        x_cubes2 = self.unfold_to_cubes(h1_voxel, self.cube_size, self.stride)
        h2 = self.encoder2(x_cubes2)
        h2_voxel = self.fold_to_voxels(h2, cur_bs, *ncubes_per_dim)

        quant2_ = self.quant_conv2(h2)

        if before_quant:
            quant2_voxel_ = self.fold_to_voxels(quant2_, cur_bs, *ncubes_per_dim)

        quant2, diff_t, id_t = self.quantize2(quant2_)
        quant2_voxel = self.fold_to_voxels(quant2, cur_bs, *ncubes_per_dim)

        dec_t, occ_l1 = self.decoder2(self.post_quant_conv2(quant2_voxel))
        dec_t_cubes = self.unfold_to_cubes(dec_t, 1, 1)
        enc_b = torch.cat([dec_t_cubes, h1], 1)

        quant1_ = self.quant_conv1(enc_b)
        cur_bs = input_data.shape[0]
        ncubes_per_dim = [i // self.cube_size for i in input_data.shape[2:]]

        if before_quant:
            quant1_voxel_ = self.fold_to_voxels(quant1_, cur_bs, *ncubes_per_dim)

        quant1, diff_b, id_b = self.quantize1(quant1_)
        quant1_voxel = self.fold_to_voxels(quant1, cur_bs, *ncubes_per_dim)
        if before_quant:
            return quant2_voxel_, quant1_voxel_, quant2_voxel, quant1_voxel, occ_l1
        else:
            return quant2_voxel, quant1_voxel, diff_t + diff_b, id_t, id_b, occ_l1

    def forward(self, data_dict):
        self.temperature_scheduling(data_dict["global_step"])

        input_data = data_dict["input"]
        quant1, quant2, diff, _, _, occ_l1 = self.patch_encode(input_data)
        dec, occ_l2 = self.decode(quant1, quant2, occ_l1)
        recon = {
            "tsdf": dec,
            "occ_l1": occ_l1,
            "occ_l2": occ_l2,
        }
        return recon, diff

    def get_code(self, vol):
        return self.patch_encode(vol, before_quant=True)

    def get_last_layer(self):
        return self.decoder1.conv_out.weight

    def get_ae_paras(self):
        paras = (
            list(self.encoder1.parameters())
            + list(self.encoder2.parameters())
            + list(self.decoder1.parameters())
            + list(self.decoder2.parameters())
            + list(self.upsampler.parameters())
            + list(self.quantize1.parameters())
            + list(self.quant_conv1.parameters())
            + list(self.post_quant_conv1.parameters())
            + list(self.quantize2.parameters())
            + list(self.quant_conv2.parameters())
            + list(self.post_quant_conv2.parameters())
        )
        return paras
