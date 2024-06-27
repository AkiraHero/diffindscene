import torch
import os
import torch.nn as nn
import torch.nn.functional as F

import logging
import time
from tqdm import tqdm
from contextlib import contextmanager
from einops import rearrange, repeat

import factory.model_factory as ModelFactory
from model.model_base import ModelBase
from model.ms_ldm.unet_model import UNetModel
from model.utils.ema import LitEma
from model.utils.global_mapper import GlobalMapper
from model.ms_ldm.sketch_encoder import SketchEncoder
from utils.config.Configuration import default

from ..diffuser.pipelines.pipeline_ddim import DDIMPipeline
from ..diffuser.schedulers.scheduling_ddim import DDIMScheduler
from ..diffuser.pipelines.pipeline_ddpm import DDPMPipeline
from ..diffuser.schedulers.scheduling_ddpm import DDPMScheduler


class MSLDM(ModelBase):
    unet_alternative = {"UNetModel": UNetModel}

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.voxel_size = 0.04

        proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.ckpt_path = os.path.join(proj_dir, 'ckpt')
        self.use_sketch_condition = default(config.paras, "use_sketch_condition", False)
        self.use_sketch_attention = default(config.paras, "use_sketch_attention", False)

        self.restore_batch_size = config.paras.multi_restore_batch_size
        self.first_stage_module = self.init_first_stage_module(
            config.paras.first_stage_model
        )

        if self.use_sketch_condition:
            self.sketch_encoder = SketchEncoder()  # no use
            smodel = ModelFactory.ModelFactory.get_model(config.paras.sketch_embedder)
            ckpt_ = config.paras.sketch_embedder.ckpt
            if not os.path.isabs(ckpt_):
                ckpt_ = os.path.join(self.ckpt_path, ckpt_)
            smodel.load_model_paras_from_file(ckpt_)
            smodel = smodel.eval()
            self.sketch_embedder = [smodel]
            self.sketch_emb_mapper = nn.Conv2d(16, 4, 1, 1, device=self.device)

        if self.use_sketch_attention:
            self.sketch_attd_emb_mapper = nn.Conv2d(16, 256, 1, 1, device=self.device)

        self.model_paras = config.paras.unet_model

        self.operating_size = {
            "first": [64, 64, 16],
            "second": [64, 64, 32],
            "third": [256, 256, 128],
        }
        self.scheduler_config = config.paras.noise_schedule

        with torch.no_grad():
            upsampler = torch.nn.ConvTranspose3d(1, 1, 2, 2)
            upsampler.weight *= 0
            upsampler.weight += 1
            upsampler.bias *= 0
            self.occ_upsampler = upsampler

        if "mode" in config and config.mode == "testing":
            self.level_list = ["first", "second", "third"]
            self.model_level = self.level_list[0]
            self.cascaded_output = {
                "first": {
                    "result": None,
                    "next_occ": None,
                },
                "second": {
                    "result": None,
                    "next_occ": None,
                },
                "third": {
                    "result": None,
                    "next_occ": None,
                },
            }
        else:
            self.model_level = config.paras.level
            self.init_level(config.paras.level)

            # add ema support
            self.use_ema = default(config.paras, "use_ema", False)
            if self.use_ema:
                self.model_ema = LitEma(self.model)
                print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # for third level: whether use all features from  previous levels
        self.use_all_feature_in_final_level = True

        # add noise to occupancy condition for 2nd/3rd level (use occ ground truth for training)
        self.add_noise_da = default(config.paras, "add_noise_da", False)

        # classifier free guidance for 2nd / 3rd conditional diffusion
        self.guidance_scale = 0.5
        self.classifier_free_guidance = default(
            config.paras, "classifier_free_guidance", False
        )

        # latent normalization
        self.enable_normalize_latent_code = True
        self.latent_stats = None
        self.register_buffer("scale_factor_code1", torch.tensor(1.0))
        self.register_buffer("scale_factor_code2", torch.tensor(1.0))

        # code of empty voxel
        self.empty_code1 = None

    def set_operating_size(self, size_first):
        self.operating_size = {
            "first": size_first,
            "second": [i * 2 for i in size_first],
            "third": [i * 8 for i in size_first],
        }

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @torch.no_grad()
    def validation_step(self, batch):
        # self.scale_factor = self.scale_factor.unsqueeze(0)
        loss_dict_no_ema = self.forward_generative(batch)
        # print(loss_dict_no_ema)
        with self.ema_scope():
            loss_dict_ema = self.forward_generative(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        return loss_dict_no_ema, loss_dict_ema

    def init_level(self, level, mode="train"):
        logging.info("Initing.... Model level={}, mode={}".format(level, mode))
        self.model_level = level
        model_type = self.model_paras[self.model_level]["model_type"]
        model_args = self.model_paras[self.model_level]["model_args"]
        self.model = self.unet_alternative[model_type](**model_args).to(self.device)

        if level == "first" and hasattr(self, "sketch_embedder"):
            self.sketch_embedder[0] = self.sketch_embedder[0].to(self.device)

        if mode == "test":
            if "ckpt" in self.model_paras[self.model_level]:
                ckpt_ = self.model_paras[self.model_level]["ckpt"]
                if not os.path.isabs(ckpt_):
                    ckpt_ = os.path.join(self.ckpt_path, ckpt_)
                ckpt_content = torch.load(ckpt_)
                pop_keys = []
                for key in ckpt_content["model_paras"].keys():
                    if key.startswith("model_ema"):
                        pop_keys += [key]
                for key in pop_keys:
                    ckpt_content["model_paras"].pop(key)
                self.load_model_paras(ckpt_content)
                logging.info("[LDM]load model:{}".format(ckpt_))
                self.model = self.model.eval()

                method = "ddim"
                self.current_scheduler_paras = self.scheduler_config[self.model_level]

            self.restoration_config = {
                "first": dict(
                    eta=1.0,
                    step_num=200,
                    use_clipped_model_output=self.current_scheduler_paras.clip_sample,
                ),
                "second": dict(
                    eta=1.0,
                    step_num=200,
                    use_clipped_model_output=self.current_scheduler_paras.clip_sample,
                ),
                "third": dict(
                    eta=1.0,
                    step_num=200,
                    use_clipped_model_output=self.current_scheduler_paras.clip_sample,
                ),
            }
        else:
            method = "ddim"
            self.current_scheduler_paras = self.scheduler_config[self.model_level]

        if method == "ddim":
            self.scheduler = DDIMScheduler(**self.current_scheduler_paras)
            self.pipeline = DDIMPipeline(self.model, self.scheduler)
        elif method == "ddpm":
            self.scheduler = DDPMScheduler(**self.current_scheduler_paras)
            self.pipeline = DDPMPipeline(self.model, self.scheduler)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def init_first_stage_module(self, config):
        model = ModelFactory.ModelFactory.get_model(config)
        ckpt_ = config.ckpt
        if not os.path.isabs(ckpt_):
            ckpt_ = os.path.join(self.ckpt_path, ckpt_)
        model.load_model_paras_from_file(ckpt_)
        model = model.eval()
        return model

    def get_empty_code(self):
        empty_tsdf = torch.ones([1, 1, 16, 16, 16]).to(self.device)
        code1, code2, _ = self.get_first_stage_code(empty_tsdf, cube_size_=16)

        code1 = code1.permute(0, 2, 3, 4, 1)
        code2 = code2.permute(0, 2, 3, 4, 1)
        return code1[0, 0, 0, 0], code2[0, 0, 0, 0]

    @staticmethod
    def chunkable(tensorshape, chunkstride):
        unchunkable = False
        for i in tensorshape:
            unchunkable |= bool(i % chunkstride)
        return not unchunkable

    @staticmethod
    def round_vol(vol, default_value, padded_shape, pad_origin=None):
        original_shape = vol.shape
        B = vol.shape[0]
        C = vol.shape[1]
        if pad_origin is None:
            pad_widths = [0, 0, 0]
        else:
            pad_widths = pad_origin

        default_value = torch.tensor(default_value, dtype=vol.dtype, device=vol.device)
        padded_volume = torch.full(
            (B, C, padded_shape[0], padded_shape[1], padded_shape[2]),
            default_value,
            device=vol.device,
        )
        padded_volume[
            :,
            :,
            pad_widths[0] : pad_widths[0] + original_shape[2],
            pad_widths[1] : pad_widths[1] + original_shape[3],
            pad_widths[2] : pad_widths[2] + original_shape[4],
        ] = vol
        return padded_volume

    @staticmethod
    def roundtensorsize(vol, round_stride=16):
        dim = list(vol.shape[2:])
        round_dim = [round_stride * ((i - 1) // round_stride + 1) for i in dim]
        roundedvol = MSLDM.round_vol(vol, 0, round_dim)
        return roundedvol, dim

    # return occ score before sigmoid
    def decode_occ_lv1(self, result, stride=16):
        if self.first_stage_module.patch_mode:
            cur_bs = result.shape[0]
            ncubes_per_dim = result.shape[2:]
            quant2_ = self.unfold_to_cubes(result, 1, 1)
            quant2, diff_t, id_t = self.first_stage_module.quantize2(quant2_)
            quant2_voxel = self.fold_to_voxels(quant2, cur_bs, *ncubes_per_dim)
        else:
            quant2_voxel, diff_t, id_t = self.first_stage_module.quantize2(result)

        enable_chunk = max(result.shape[2:]) > 32 and self.chunkable(
            result.shape[2:], stride
        )
        if enable_chunk:
            cur_bs = quant2_voxel.shape[0]
            ncubes_per_dim = [i // stride for i in quant2_voxel.shape[2:]]
            quant2_voxel_cube = self.unfold_to_cubes(quant2_voxel, stride, stride)
            dec_t_list = []
            occ_l1_list = []
            for i in range(quant2_voxel_cube.shape[0]):
                dec_t_, occ_l1_ = self.first_stage_module.decoder2(
                    self.first_stage_module.post_quant_conv2(
                        quant2_voxel_cube[i : i + 1]
                    )
                )
                dec_t_list += [dec_t_]
                occ_l1_list += [occ_l1_]
            dec_t = torch.cat(dec_t_list, dim=0)
            occ_l1 = torch.cat(occ_l1_list, dim=0)
            dec_t = self.fold_to_voxels(dec_t, cur_bs, *ncubes_per_dim)
            occ_l1 = self.fold_to_voxels(occ_l1, cur_bs, *ncubes_per_dim)
        else:
            dec_t, occ_l1 = self.first_stage_module.decoder2(
                self.first_stage_module.post_quant_conv2(quant2_voxel)
            )
        return occ_l1, quant2_voxel

    def decode_occ_lv2(self, result, occ_l1, quant2_voxel):

        cur_bs = result.shape[0]
        assert cur_bs == 1
        ncubes_per_dim = result.shape[2:]

        quant1_ = self.unfold_to_cubes(result, 1, 1)
        if quant1_.shape[0] > 64 * 64 * 32:
            mini_batch_size = 64 * 64 * 32
            num_samples = quant1_.shape[0]
            num_batches = num_samples // mini_batch_size
            q_batch_list = []
            for i in range(num_batches):
                start_idx = i * mini_batch_size
                end_idx = (i + 1) * mini_batch_size
                batch_a = quant1_[start_idx:end_idx]
                q_batch, _, _ = self.first_stage_module.quantize1(batch_a)
                q_batch_list += [q_batch]
            remaining_samples = num_samples % mini_batch_size
            if remaining_samples > 0:
                start_idx = num_batches * mini_batch_size
                end_idx = num_samples
                batch_a = quant1_[start_idx:end_idx]
                q_batch, _, _ = self.first_stage_module.quantize1(batch_a)
                q_batch_list += [q_batch]
            quant1 = torch.cat(q_batch_list, dim=0)
        else:

            quant1, diff_b, id_b = self.first_stage_module.quantize1(quant1_)
        quant1_voxel = self.fold_to_voxels(quant1, cur_bs, *ncubes_per_dim)

        enable_chunk = max(result.shape[2:]) > 48
        if enable_chunk:

            quant2_voxel_, original_dim = self.roundtensorsize(
                quant2_voxel, round_stride=16
            )
            quant1_voxel_ = self.round_vol(
                quant1_voxel, 0, [i * 2 for i in quant2_voxel_.shape[2:]]
            )
            occ_l1_ = self.round_vol(
                occ_l1, 0, [i * 2 for i in quant2_voxel_.shape[2:]]
            )

            ncubes_per_dim = [i // 16 for i in quant2_voxel_.shape[2:]]
            cur_bs = quant2_voxel.shape[0]
            quant2_voxel_minibatch = self.unfold_to_cubes(quant2_voxel_, 16, 16)
            quant1_voxel_minibatch = self.unfold_to_cubes(quant1_voxel_, 32, 32)
            occl1_voxel_minibatch = self.unfold_to_cubes(occ_l1_, 32, 32)
            occ_l2_list = []
            for i in range(quant2_voxel_minibatch.shape[0]):
                dec, occ_l2 = self.first_stage_module.decode(
                    quant2_voxel_minibatch[i : i + 1],
                    quant1_voxel_minibatch[i : i + 1],
                    occlv1=occl1_voxel_minibatch[i : i + 1],
                    sparse_decode=True,
                )
                occ_l2_list += [occ_l2]
            occ_l2 = torch.cat(occ_l2_list, dim=0)
            occ_l2 = self.fold_to_voxels(occ_l2, cur_bs, *ncubes_per_dim)
            occ_l2_original_dim = [8 * i for i in original_dim]
            return occ_l2[
                :,
                :,
                : occ_l2_original_dim[0],
                : occ_l2_original_dim[1],
                : occ_l2_original_dim[2],
            ]
        else:
            dec, occ_l2 = self.first_stage_module.decode(
                quant2_voxel, quant1_voxel, occ_l1, sparse_decode=True
            )
            return occ_l2

    @torch.no_grad()
    def restoration(self, datadict):
        for level in self.level_list:
            self.init_level(level, mode="test")
            if level == "first":
                st_time = time.time()
                result, ini_mask = self.restoration_cascaded_general(
                    default_grid_value=None,
                    datadict=datadict,
                    **self.restoration_config[level],
                )
                ed_time = time.time()

                scene_size = None
                if self.use_sketch_condition:
                    assert result.shape[0] == 1
                    max_xyz, _ = ini_mask[0][0].nonzero().max(dim=0)
                    min_xyz, _ = ini_mask[0][0].nonzero().min(dim=0)
                    result = result[
                        :,
                        :,
                        min_xyz[0] : max_xyz[0] + 1,
                        min_xyz[1] : max_xyz[1] + 1,
                        min_xyz[2] : max_xyz[2] + 1,
                    ]
                    scene_size = max_xyz - min_xyz

                self.cascaded_output[level]["result"] = result

                if self.enable_normalize_latent_code:
                    # result = self.denormalize_latent(result, self.latent_stats['code_lv2'])
                    result = result / self.scale_factor_code1
                occ_l1, quant2_voxel = self.decode_occ_lv1(result)

                occ = torch.sigmoid(occ_l1)
                thres = 0.5
                occ_mask = occ > thres
                occ[occ_mask] = 1.0
                occ[~occ_mask] = 0.0
                self.cascaded_output[level]["next_occ"] = occ.to(torch.bool)
                self.cascaded_output[level]["info"] = {
                    "occ_l1": occ_l1,
                    "quant2_voxel": quant2_voxel,
                    "ini_mask": ini_mask,
                    "time_consumption": ed_time - st_time,
                }
                if scene_size is not None:
                    self.cascaded_output[level]["info"].update({"scene_size": scene_size})
                if self.use_sketch_condition:
                    self.cascaded_output[level]["info"].update({"sketch": datadict["bev_sketch"]})

            elif level == "second":
                st_time = time.time()
                result, _ = self.restoration_cascaded_general(
                    default_grid_value=0.0, **self.restoration_config[level]
                )
                ed_time = time.time()

                self.cascaded_output[level]["result"] = result

                if self.enable_normalize_latent_code:
                    result = result / self.scale_factor_code2
                occ_l1 = self.cascaded_output["first"]["info"]["occ_l1"]
                quant2_voxel = self.cascaded_output["first"]["info"]["quant2_voxel"]

                occ_l2 = self.decode_occ_lv2(result, occ_l1, quant2_voxel)
                occ_prior = self.occ_upsampler(
                    self.occ_upsampler(
                        self.cascaded_output["first"]["next_occ"].to(torch.float)
                    )
                )
                occ_prior = occ_prior.to(torch.bool)

                occ = torch.sigmoid(occ_l2)
                thres = 0.5
                occ_mask = occ > thres
                occ[occ_mask] = 1.0
                occ[~occ_mask] = 0.0
                occ[~occ_prior] = 0

                self.cascaded_output[level]["next_occ"] = occ.to(torch.bool)
                self.cascaded_output[level]["info"] = {
                    "time_consumption": ed_time - st_time,
                }

            elif level == "third":
                st_time = time.time()
                result = self.restoration_cascaded_multi(
                    default_grid_value=1.0, **self.restoration_config[level]
                )
                ed_time = time.time()
                self.cascaded_output[level]["result"] = result
                self.cascaded_output[level]["info"] = {
                    "time_consumption": ed_time - st_time,
                }
            else:
                raise NotImplementedError
        return self.cascaded_output

    def restoration_cascaded_general(
        self,
        default_grid_value=0.0,
        eta=0.9,
        step_num=200,
        use_clipped_model_output=False,
        datadict=None,
    ):
        test_input = self.get_test_input(datadict=datadict)

        if datadict is not None:
            start_time_step_inx = (
                datadict["start_time_step_inx"]
                if "start_time_step_inx" in datadict
                else 0
            )
        else:
            start_time_step_inx = 0

        y_cond = test_input["condition"]
        y_t = test_input["y_t"]
        bool_mask = test_input["mask"]
        completion_mask = bool_mask
        cond_data = None
        if "attd_emb" in test_input:
            cond_data = test_input["attd_emb"]

        output = self.pipeline(
            y_cond,
            y_t,
            mask=bool_mask,
            num_inference_steps=step_num,
            use_clipped_model_output=use_clipped_model_output,
            eta=eta,
            txt_cond=cond_data,
            start_time_step_inx=start_time_step_inx,
        )

        output_ = output.permute(0, 2, 3, 4, 1)
        bool_mask_ = bool_mask.squeeze(1)
        if default_grid_value is not None:
            output_[~bool_mask_] = default_grid_value
        output_ = output_.permute(0, 4, 1, 2, 3)
        return output, bool_mask

    def forward(self, datadict):
        return self.forward_generative(datadict)

    @staticmethod
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

    @staticmethod
    def generate_coordinate_volume(volume_shape):
        assert 3 == len(volume_shape)
        coordinates = torch.meshgrid(
            torch.arange(volume_shape[0]),
            torch.arange(volume_shape[1]),
            torch.arange(volume_shape[2]),
        )
        coordinates = torch.stack(coordinates, dim=-1)
        return coordinates

    @torch.no_grad()
    def get_first_stage_code(self, tsdf_vol, cube_size_=96, decode=False):
        batch_size = tsdf_vol.shape[0]
        stride = cube_size = cube_size_
        ncubes = [i // cube_size for i in tsdf_vol.shape[2:]]
        minibatch = self.unfold_to_cubes(tsdf_vol, cube_size, stride)
        minibatch_size = 4
        code1_list = []
        code2_list = []
        code1q_list = []
        code2q_list = []
        occl1_list = []

        chunk_num = minibatch.shape[0] // minibatch_size
        if chunk_num > 0:
            assert minibatch.shape[0] % minibatch_size == 0
            chunks = torch.chunk(minibatch, chunk_num)
        else:
            chunks = [minibatch]

        for piece in chunks:

            code1, code2, code1_q, code2_q, occl1 = self.first_stage_module.get_code(
                piece
            )

            code1_list += [code1]
            code2_list += [code2]
            code1q_list += [code1_q]
            code2q_list += [code2_q]
            occl1_list += [occl1]

        code1 = torch.cat(code1_list, dim=0)
        code2 = torch.cat(code2_list, dim=0)
        code1q = torch.cat(code1q_list, dim=0)
        code2q = torch.cat(code2q_list, dim=0)
        occl1 = torch.cat(occl1_list, dim=0)
        code1 = self.fold_to_voxels(code1, batch_size, *ncubes)
        code2 = self.fold_to_voxels(code2, batch_size, *ncubes)
        code1q = self.fold_to_voxels(code1q, batch_size, *ncubes)
        code2q = self.fold_to_voxels(code2q, batch_size, *ncubes)
        occl1 = self.fold_to_voxels(occl1, batch_size, *ncubes)
        if decode:
            dec, occl2 = self.first_stage_module.decode(code1q, code2q)
            return code1, code2, occl1, occl2
        return code1, code2, occl1

    @staticmethod
    def add_noise_to_binary_occ(occ_vol, noise_scale):
        # only add noise to occ section, keep the empty grids empty as before
        dtype = occ_vol.dtype
        assert 0 <= noise_scale <= 1
        occ_vol = occ_vol.to(torch.bool)
        empty_vol = ~occ_vol
        # Generate random values between 0 and 1
        noise = torch.rand(occ_vol.shape)
        binary_noise = noise < noise_scale
        occ_vol[binary_noise] = ~(occ_vol[binary_noise])
        occ_vol[empty_vol] = False
        return occ_vol.to(dtype)

    @torch.no_grad()
    def prepare_training(self, datadict):
        if self.enable_normalize_latent_code:
            del self.scale_factor_code1
            del self.scale_factor_code2
            code1_scale = datadict["latent_scale"][0][0]
            code2_scale = datadict["latent_scale"][0][1]
            self.register_buffer("scale_factor_code1", code1_scale)
            logging.info(
                "[register_buffer]scale_factor_code1={}".format(self.scale_factor_code1)
            )
            self.register_buffer("scale_factor_code2", code2_scale)
            logging.info(
                "[register_buffer]scale_factor_code2={}".format(self.scale_factor_code2)
            )

    def get_txt_emb(self, des):
        if isinstance(self.text_embedder, list):
            text_embedder = self.text_embedder[0]
        else:
            text_embedder = self.text_embedder
        text_embedder = text_embedder.to(self.device)

        des_list_cat = []
        for i in des:
            des_list_cat += i
        text_embedding = text_embedder(des_list_cat)

        split_tensors = text_embedding.view(-1, 2, 77, 768)
        reshaped_tensors = split_tensors.view(-1, 1, 154, 768)
        text_embedding = reshaped_tensors.view(-1, 154, 768)
        if self.use_text_emb_mapper:
            text_embedding = self.text_emb_mapper(text_embedding)
        return text_embedding

    # @torch.no_grad()
    def get_input(self, datadict):
        new_datadict = {}

        if self.use_sketch_condition:
            sketch_embedding_vae_ = self.sketch_embedder[0].get_code(
                datadict["bev_sketch"].squeeze(-1)
            )
            sketch_embedding_vae = self.sketch_emb_mapper(sketch_embedding_vae_)
        sketch_embedding_vae_attention = None
        if self.use_sketch_attention:
            sketch_embedding_vae_attention = self.sketch_attd_emb_mapper(
                sketch_embedding_vae_
            )
            sketch_embedding_vae_attention = sketch_embedding_vae_attention.permute(
                0, 2, 3, 1
            ).contiguous()

            context_dict = {"type": "sketch_emb", "emb": sketch_embedding_vae_attention}

        if "gt_tsdf" in datadict:
            mode = "tsdf_mode"
        elif "padded_code1" in datadict:
            mode = "latent_mode"
        else:
            raise NotImplementedError

        latent_dim = 4
        scale_factor = 4

        with torch.no_grad():
            upsampler = torch.nn.ConvTranspose3d(
                latent_dim, latent_dim, scale_factor, scale_factor, device=self.device
            )
            upsampler.weight *= 0
            upsampler.weight += 1
            upsampler.bias *= 0

            upsamplerx2 = torch.nn.ConvTranspose3d(
                latent_dim, latent_dim, 2, 2, device=self.device
            )
            upsamplerx2.weight *= 0
            upsamplerx2.weight += 1
            upsamplerx2.bias *= 0

            upsamplerx4c8 = torch.nn.ConvTranspose3d(8, 8, 4, 4, device=self.device)
            upsamplerx4c8.weight *= 0
            upsamplerx4c8.weight += 1
            upsamplerx4c8.bias *= 0

        if self.model_level == "first":
            if mode == "tsdf_mode":
                tsdf_vol = datadict["gt_tsdf"][-1]
                st = time.time()
                code1, code2, occl1_model_score = self.get_first_stage_code(
                    tsdf_vol, cube_size_=tsdf_vol.shape[-1]
                )
                ed = time.time()
                print("get_first_stage_code:", ed - st)
            elif mode == "latent_mode":
                code1, code2, occl1_model_score = (
                    datadict["padded_code1"],
                    datadict["padded_code2"],
                    datadict["padded_occl1"],
                )
                code1_mask = datadict["code1_mask"]
                code2_mask = datadict["code2_mask"]
                if not self.model.is_sparse:
                    if self.empty_code1 is None:
                        # get empty code:
                        self.empty_code1, _ = self.get_empty_code()
                    empty_mask = ~code1_mask[:, 0, ...]
                    code1_tmp = code1.permute(0, 2, 3, 4, 1)
                    code1_tmp[empty_mask] = self.empty_code1.reshape(1, 4)
                    code1 = code1_tmp.permute(0, 4, 1, 2, 3)
            else:
                raise NotImplementedError

            if self.enable_normalize_latent_code:
                code1 *= self.scale_factor_code1
                code2 *= self.scale_factor_code2

            resized_sketch_volume = sketch_embedding_vae.unsqueeze(-1).repeat(
                [1, 1, 1, 1, 16]
            )

            new_datadict = {
                "gt": code1,
                "mask": torch.ones(code1.shape[2:])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(code1.shape[0], 1, 1, 1, 1)
                .to(torch.bool)
                .to(code1.device),
                "condition": resized_sketch_volume,
            }
            if mode == "latent_mode":
                new_datadict.update(
                    {
                        "mask": code1_mask[:, 0:1, :, :, :],
                    }
                )
            if not self.model.is_sparse:
                new_datadict["mask"] |= True
                # new_datadict['weight_mask'] = code1_mask
            if sketch_embedding_vae_attention is not None:
                new_datadict.update(dict(attd_emb=context_dict))
            if self.use_sketch_condition:
                new_datadict.update(dict(sketch_emb=sketch_embedding_vae))

        elif self.model_level == "second":
            if mode == "tsdf_mode":
                tsdf_vol = datadict["gt_tsdf"][-1]
                st = time.time()
                code1, code2, occl1_model_score = self.get_first_stage_code(
                    tsdf_vol, cube_size_=tsdf_vol.shape[-1]
                )
                ed = time.time()
                print("get_first_stage_code:", ed - st)
            elif mode == "latent_mode":
                code1, code2, occl1_model_score = (
                    datadict["padded_code1"],
                    datadict["padded_code2"],
                    datadict["padded_occl1"],
                )
                code1_mask = datadict["code1_mask"]
                code2_mask = datadict["code2_mask"]
            else:
                raise NotImplementedError

            # guidance
            self.guidance_scale = 0.3

            if self.enable_normalize_latent_code:
                code1 *= self.scale_factor_code1
                code2 *= self.scale_factor_code2

            model_occ_mask = torch.sigmoid(occl1_model_score) > 0.5
            model_occ = torch.zeros_like(occl1_model_score)
            model_occ[model_occ_mask] = 1.0
            if self.add_noise_da:
                model_occ = self.add_noise_to_binary_occ(model_occ, 0.02)
                noise_added = torch.randn_like(code1) * 0.1
                code_used = code1 + noise_added
                occ_used = model_occ.to(torch.bool)
            else:
                code_used = code1
                occ_used = model_occ.to(torch.bool)
            new_datadict = {
                "gt": code2,
                "mask": occ_used,  # todo: use occ from code1 decode
                "condition": upsamplerx2(code_used),
            }

            if self.classifier_free_guidance and torch.rand(1) < self.guidance_scale:
                new_datadict["condition"] = torch.randn_like(new_datadict["condition"])

        elif self.model_level == "third":
            if mode == "tsdf_mode":
                tsdf_vol = datadict["gt_tsdf"][-1]
                st = time.time()
                code1, code2, occl1_model_score = self.get_first_stage_code(
                    tsdf_vol, cube_size_=tsdf_vol.shape[-1]
                )
                ed = time.time()
                print("get_first_stage_code:", ed - st)
            elif mode == "latent_mode":
                code1, code2, occl1_model_score, tsdf_vol = (
                    datadict["padded_code1"],
                    datadict["padded_code2"],
                    datadict["padded_occl1"],
                    datadict["padded_tsdf"],
                )
            else:
                raise NotImplementedError

            # guidance
            self.guidance_scale = 0.3

            if self.enable_normalize_latent_code:
                code1 *= self.scale_factor_code1
                code2 *= self.scale_factor_code2

            if self.use_all_feature_in_final_level:
                code2 = torch.cat([upsamplerx2(code1), code2], dim=1)

            occ_lv2_gt = (tsdf_vol.abs() < 1.0).to(torch.float)

            if self.add_noise_da:
                occ_lv2_noisy = self.add_noise_to_binary_occ(occ_lv2_gt, 0.05)
                noise_added = torch.randn_like(code2) * 0.1
                code_used = code2 + noise_added
                occ_used = occ_lv2_noisy.to(torch.bool)
            else:
                code_used = code2
                occ_used = occ_lv2_gt.to(torch.bool)
            new_datadict = {
                "gt": tsdf_vol,
                "mask": occ_used,  # todo: use occ from code2 decode
                "condition": (
                    upsampler(code_used)
                    if code_used.shape[1] == 4
                    else upsamplerx4c8(code_used)
                ),
            }
            if self.classifier_free_guidance and torch.rand(1) < self.guidance_scale:
                new_datadict["condition"] = torch.randn_like(new_datadict["condition"])

        else:
            raise NotImplementedError
        return new_datadict

    @torch.no_grad()
    def get_test_input(self, datadict=None, batch_size=1):
        latent_dim = 4
        scale_factor = 4
        upsampler = torch.nn.ConvTranspose3d(
            latent_dim, latent_dim, scale_factor, scale_factor, device=self.device
        )
        upsampler.weight *= 0
        upsampler.weight += 1
        upsampler.bias *= 0

        upsamplerx2 = torch.nn.ConvTranspose3d(
            latent_dim, latent_dim, 2, 2, device=self.device
        )
        upsamplerx2.weight *= 0
        upsamplerx2.weight += 1
        upsamplerx2.bias *= 0

        upsamplerx4c8 = torch.nn.ConvTranspose3d(8, 8, 4, 4, device=self.device)
        upsamplerx4c8.weight *= 0
        upsamplerx4c8.weight += 1
        upsamplerx4c8.bias *= 0

        if self.model_level == "first":
            # code_size = code1.shape[2:]
            input_size = [batch_size, latent_dim] + self.operating_size[
                self.model_level
            ]
            mask_size = [batch_size, 1] + self.operating_size[self.model_level]
            new_datadict = {
                "y_t": (
                    torch.randn(input_size).to(self.device)
                    if "y_t" not in datadict
                    else datadict["y_t"]
                ),
                "mask": torch.ones(mask_size).to(torch.bool).to(self.device),
                "condition": None,
            }
            # if text_embedding is not None:
            #     new_datadict.update(dict(text_emb=text_embedding))

            if self.use_sketch_condition:
                sketch_embedding_vae_ = self.sketch_embedder[0].get_code(
                    datadict["bev_sketch"].squeeze(-1)
                )
                sketch_embedding_vae = self.sketch_emb_mapper(sketch_embedding_vae_)
                sketch_embedding_vae_attention = self.sketch_attd_emb_mapper(
                    sketch_embedding_vae_
                )
                sketch_embedding_vae_attention = sketch_embedding_vae_attention.permute(
                    0, 2, 3, 1
                ).contiguous()

                context_dict = {
                    "type": "sketch_emb",
                    "emb": sketch_embedding_vae_attention,
                }
                resized_sketch_volume = sketch_embedding_vae.unsqueeze(-1).repeat(
                    [1, 1, 1, 1, 16]
                )
                new_datadict.update(dict(condition=resized_sketch_volume[0:1]))
                if sketch_embedding_vae_attention is not None:
                    new_datadict.update(dict(attd_emb=context_dict))

                # test code mask
                code1_mask = datadict["code1_mask"]
                new_datadict.update(
                    {
                        "mask": code1_mask[0:1, 0:1, :, :, :],
                    }
                )

        elif self.model_level == "second":
            input_size = [batch_size, latent_dim] + self.operating_size[
                self.model_level
            ]
            mask_size = [batch_size, 1] + self.operating_size[self.model_level]
            if self.cascaded_output["first"]["next_occ"] is not None:
                input_size = [batch_size, latent_dim] + list(
                    self.cascaded_output["first"]["next_occ"].shape[2:]
                )
                mask_size = [batch_size, 1] + list(
                    self.cascaded_output["first"]["next_occ"].shape[2:]
                )
            new_datadict = {
                "y_t": torch.randn(input_size).to(self.device),
                "mask": self.cascaded_output["first"]["next_occ"],
                "condition": upsamplerx2(self.cascaded_output["first"]["result"]),
            }

        elif self.model_level == "third":
            input_size = [batch_size, 1] + self.operating_size[self.model_level]
            if self.cascaded_output["second"]["next_occ"] is not None:
                input_size = [batch_size, 1] + list(
                    self.cascaded_output["second"]["next_occ"].shape[2:]
                )

            if self.use_all_feature_in_final_level:
                code = torch.cat(
                    [
                        upsamplerx2(self.cascaded_output["first"]["result"]),
                        self.cascaded_output["second"]["result"],
                    ],
                    dim=1,
                )
            else:
                code = self.cascaded_output["second"]["result"]

            new_datadict = {
                "y_t": torch.randn(input_size).to(self.device),
                "mask": self.cascaded_output["second"]["next_occ"],
                "condition": (
                    upsampler(code) if code.shape[1] == 4 else upsamplerx4c8(code)
                ),
            }

        else:
            raise NotImplementedError
        return new_datadict

    def forward_generative(self, datadict):
        new_data_dict = self.get_input(datadict)
        gt_image = new_data_dict["gt"]
        bool_mask = new_data_dict["mask"]

        # repeat mask for channels
        bool_mask_full = bool_mask.repeat(1, gt_image.shape[1], 1, 1, 1)
        mask_full = bool_mask_full.to(torch.int)

        ycond = new_data_dict["condition"]

        # Sample noise to add to the images
        noise = torch.randn_like(gt_image)
        bs = gt_image.shape[0]

        # Sample a random timestep for each image
        timestep_max = self.scheduler.num_train_timesteps
        timestep_min = 1

        timesteps = torch.randint(
            timestep_min, timestep_max, (bs,), device=gt_image.device
        ).long()
        timesteps = timesteps.reshape(bs, 1)

        assert not gt_image.isnan().any()
        y_noisy, gammas = self.scheduler.add_noise(gt_image, noise, timesteps)

        blank = torch.zeros_like(y_noisy)

        context = None
        # classifier free guidance
        if torch.rand(()) < 0.1:
            context = None
            ycond = torch.randn_like(ycond)
        else:
            if "attd_emb" in new_data_dict:
                context = new_data_dict["attd_emb"]

        if ycond is not None:
            cond_input = torch.cat(
                [ycond, y_noisy * mask_full + (1.0 - mask_full) * blank], dim=1
            )
        else:
            cond_input = y_noisy * mask_full + (1.0 - mask_full) * blank

        noise_pred = self.model(cond_input, gammas, mask=bool_mask, context=context)

        mse_loss = F.mse_loss(noise_pred, noise, reduction="none")

        if "weight_mask" in new_data_dict:
            w_mask = new_data_dict["weight_mask"]
            weight = torch.ones_like(w_mask, dtype=torch.float32)
            weight[~w_mask] = 0.3
            mse_loss *= weight

        loss = mse_loss[bool_mask_full].mean()

        loss_dict = {
            "total": loss,
            "mse_loss_mat": mse_loss.detach(),
            "loss_mask": bool_mask_full.detach(),
            "timestep": timesteps.detach(),
            "gammas": gammas.detach(),
        }

        return loss_dict

    def break_info_overlapped_pieces(self, vol, cube_size, stride):
        return self.unfold_to_cubes(vol, cube_size, stride)

    def break_info_overlapped_pieces_new(self, vol, default_v, stride):
        # decide cube size
        cube_size = vol.shape[-1]
        assert cube_size == min(list(vol.shape[2:]))
        assert stride < cube_size
        # decide the padding shape
        padding_shape = [
            stride * ((i - 1 - cube_size) // stride + 1) + cube_size
            for i in vol.shape[2:]
        ]
        new_vol = self.round_vol(vol, default_v, padding_shape)
        return (
            self.unfold_to_cubes(new_vol, cube_size, stride),
            padding_shape,
            cube_size,
        )

    def restoration_cascaded_multi(
        self,
        default_grid_value=0.0,
        step_num=2000,
        eta=0.9,
        use_clipped_model_output=False,
    ):
        num_inference_steps = step_num
        use_clipped_model_output = use_clipped_model_output
        eta = eta

        test_input = self.get_test_input()
        y_cond = test_input["condition"]
        y_t = test_input["y_t"]
        bool_mask = test_input["mask"]

        # only support single scene
        assert y_t.shape[0] == 1

        scene_origin = torch.tensor([0, 0, 0]).to(self.device)
        scene_dim = list(y_t.shape)[2:]

        voxel_size = 0.04

        # piece_cube = 128
        piece_stride = 64
        if scene_dim[0] == 512:
            piece_stride = 96

        y_t, padded_scene_dim, adaptive_cube_size = (
            self.break_info_overlapped_pieces_new(y_t, 0, piece_stride)
        )
        y_cond, _, _ = self.break_info_overlapped_pieces_new(y_cond, 0, piece_stride)
        bool_mask, _, _ = self.break_info_overlapped_pieces_new(
            bool_mask, 0, piece_stride
        )

        coords = (
            self.generate_coordinate_volume(padded_scene_dim)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(self.device)
        )
        coords = self.break_info_overlapped_pieces(
            coords, adaptive_cube_size, piece_stride
        )

        image = y_t
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        bs = y_t.shape[0]

        final_map = None
        for t in tqdm(
            self.scheduler.timesteps,
            desc="sampling loop time step",
            total=num_inference_steps,
        ):

            count = GlobalMapper(
                scene_origin,
                scene_dim,
                default_value=0,
                device=image.device,
                voxel_size=voxel_size,
            )
            value = GlobalMapper(
                scene_origin,
                scene_dim,
                default_value=0,
                device=image.device,
                voxel_size=voxel_size,
            )
            # 1. predict noise model_output
            # mask done in unet...
            model_input = torch.cat([y_cond, image], dim=1)
            alphas_cumprod = self.scheduler.alphas_cumprod.to(image.device)

            input_bs = bs
            use_bs = (
                input_bs
                if input_bs < self.restore_batch_size
                else self.restore_batch_size
            )
            timestep_encoding = alphas_cumprod[t].repeat(use_bs, 1).to(y_t.device)
            chunk_size = int(torch.ceil(torch.tensor(input_bs / use_bs)).item())
            input_chunks = torch.chunk(model_input, chunk_size)
            mask_chunks = torch.chunk(bool_mask, chunk_size)
            output_chunk_list = []
            for input_chunk, mask_chunk in zip(input_chunks, mask_chunks):
                model_output_chunk = self.model(
                    input_chunk, timestep_encoding, mask=mask_chunk
                )  # .sample
                output_chunk_list.append(model_output_chunk)
            model_output = torch.cat(output_chunk_list)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image_ = self.scheduler.step(
                model_output,
                t,
                image,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=None,
            ).prev_sample

            image_[~bool_mask] = default_grid_value

            bs = image_.shape[0]
            id_list = []

            for piece_inx in range(bs):
                part_id = piece_inx
                part_vol = image_[piece_inx].squeeze(0)
                cnt_vol = (
                    bool_mask[piece_inx].squeeze(0).to(torch.int)
                )  # torch.ones_like(part_vol)
                part_origin = (
                    coords[piece_inx].permute(1, 2, 3, 0)[0, 0, 0] * voxel_size
                )
                value.update(part_id, part_vol, part_origin, mode="random")
                count.update(part_id, cnt_vol, part_origin, mode="add")
                id_list.append(part_id)
            value_map = value.get_scene_map()["map"]
            cnt_map = count.get_scene_map()["map"]
            default_value = torch.ones_like(value.scene_map)

            value.scene_map = torch.where(cnt_map > 0, value_map, default_value)
            piece_list = []
            for id_ in id_list:
                piece = value.get(id_, default_value=1.0).unsqueeze(0)
                piece_list.append(piece)
            image = torch.stack(piece_list)

            final_map = value.scene_map.clone()
        return final_map.unsqueeze(0)

    def get_trainable_parameters(self):
        paras = list(self.model.parameters())
        if self.use_sketch_condition:
            paras += list(self.sketch_encoder.parameters())
            paras += list(self.sketch_emb_mapper.parameters())
        if self.use_sketch_attention:
            paras += list(self.sketch_attd_emb_mapper.parameters())
        return paras

    def load_model_paras(self, params):
        if params is not None:
            super(ModelBase, self).load_state_dict(params["model_paras"], strict=False)
        else:
            raise AssertionError("Fail to load params for model.")

    def set_device(self, device):
        super().set_device(device)
        if hasattr(self, "sketch_embedder"):
            self.sketch_embedder[0] = self.sketch_embedder[0].to(device)
