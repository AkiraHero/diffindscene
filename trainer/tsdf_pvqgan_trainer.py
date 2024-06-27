import os
import logging
import tqdm

import torch
import torch.nn as nn

import pickle
from einops import rearrange
from skimage import measure

from trainer.trainer_base import TrainerBase
from utils.visualize_occ import occ2mesh, mesh2ply
from utils.torch_distributed_config import allreduce_grads


class TSDFPVQGANTrainer(TrainerBase):
    def __init__(self, config):
        super().__init__()
        assert config.config_type in ["training", "testing"]
        self.mode = "testing"
        if config.config_type == "training":
            self.optimizer_config = config["optimizer"]
            self.max_epoch = config["epoch"]
            self.enable_val = config.enable_val
            self.val_interval = config.val_interval

            self.mode = "training"
        elif config.config_type == "testing":
            self.save_mesh = config.save_mesh
            self.output_dir = config.output_dir

        if not self.distributed:
            self.device = torch.device(config["device"])

    def set_optimizer(self, optimizer_config):
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        lr = optimizer_config["paras"]["lr"]
        opt_ae = torch.optim.Adam(model.get_ae_paras(), lr=lr)
        opt_disc = None
        if isinstance(model.loss.discriminator, nn.Module):
            opt_disc = torch.optim.Adam(model.loss.discriminator.parameters(), lr=lr)
        self.optimizer = [opt_ae, opt_disc]

    def run(self):
        # torch.autograd.set_detect_anomaly(True)
        if not self.check_ready():
            raise ModuleNotFoundError(
                "The trainer not ready. Plz set model/dataset first"
            )
        super(TSDFPVQGANTrainer, self).run()
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        for epoch in range(self.max_epoch):
            self.is_val = False
            self.epoch += 1
            model.train()

            for step, data in enumerate(self.data_loader):
                data = self.data_loader.dataset.load_data_to_gpu(data, self.device)

                input_data = data["gt_tsdf"]
                rec_target = data["gt_tsdf"]
                voxel_size = data["voxel_size"].item()
                data_dict = {
                    "input": input_data,
                    "mask": input_data.abs() < 1.0,
                    "target": rec_target,
                    "global_step": self.global_step,
                }

                xrec, qloss = model(data_dict)

                # rec loss
                optimizer_idx = 0
                vqloss, vq_log_dict = model.loss(
                    qloss,
                    rec_target,
                    xrec,
                    optimizer_idx,
                    self.global_step,
                    last_layer=model.get_last_layer(),
                    split="train",
                )
                self.optimizer[optimizer_idx].zero_grad()
                vqloss.backward(retain_graph=True)
                if self.distributed:
                    allreduce_grads(model.parameters())
                self.optimizer[optimizer_idx].step()

                # gan loss
                optimizer_idx = 1
                ganloss, gan_log_dict = model.loss(
                    qloss,
                    rec_target,
                    xrec,
                    optimizer_idx,
                    self.global_step,
                    last_layer=model.get_last_layer(),
                    split="train",
                )
                self.optimizer[optimizer_idx].zero_grad()
                ganloss.backward(retain_graph=True)
                if self.distributed:
                    allreduce_grads(model.parameters())
                self.optimizer[optimizer_idx].step()

                # print current status and logging
                if not self.distributed or self.rank == 0:
                    # log loss dict to tensorboard
                    for k, v in vq_log_dict.items():
                        self.logger.log_data(k, v.item(), True)
                    for k, v in gan_log_dict.items():
                        self.logger.log_data(k, v.item(), True)
                    # log mesh
                    if step == 0:
                        self.log_mesh(xrec, rec_target, voxel_size)
                    total_loss = vq_log_dict["train/total_loss"]
                    logging.info(
                        f"[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t"
                        f"loss={total_loss:.6f}\t"
                    )
                self.step = step
                self.global_step += 1

            if not self.distributed or self.rank == 0:
                if self.enable_val:
                    if epoch % self.val_interval == 0:
                        self.is_val = True
                        self.val_step()

            if not self.distributed or self.rank == 0:
                self.logger.log_model_params(self.model, optimizers=self.optimizer)
        self.logger.log_model_params(self.model, optimizers=self.optimizer, force=True)

    def log_mesh(self, xrec, rec_target, voxel_size):
        gt_tsdf = rec_target[0]
        verts, faces, norms, vals = measure.marching_cubes(
            gt_tsdf.squeeze(0).cpu().numpy(), level=None
        )
        verts = verts * voxel_size

        verts = torch.tensor(verts.copy()).unsqueeze(0)
        faces = torch.tensor(faces.copy()).unsqueeze(0)
        self.logger.log_mesh("gt", verts, faces)
        if isinstance(xrec, dict):
            xrec = xrec["tsdf"]
        rec_tsdf = xrec[0].detach()

        verts, faces, norms, vals = measure.marching_cubes(
            rec_tsdf.squeeze(0).cpu().numpy(), level=None
        )
        verts = verts * voxel_size
        if verts is not None:
            verts = torch.tensor(verts.copy()).unsqueeze(0)
            faces = torch.tensor(faces.copy()).unsqueeze(0)
            self.logger.log_mesh("rec", verts, faces)

    @staticmethod
    def L1metric(input, target):
        with torch.no_grad():
            loss = nn.L1Loss()
            output = loss(input, target)
        return output

    def val_step(self):
        raise NotImplementedError

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

    # round to voxels with 32 factor
    @staticmethod
    def round_vol(vol, default_value, vol_size=64):
        original_shape = vol.shape
        B = vol.shape[0]
        C = vol.shape[1]
        padded_shape = [
            vol_size * ((s - 1) // vol_size + 1) for s in original_shape[2:]
        ]  # 计算填充后的尺寸

        pad_widths = [0, 0, 0]
        # for original_size, padded_size in zip(original_shape[2:], padded_shape):
        #     pad_width = (padded_size - original_size) // 2
        #     pad_widths.append(pad_width)

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
        return padded_volume, pad_widths

    def run_test(self):
        super(TSDFPVQGANTrainer, self).run_test()
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        self.global_step = 0
        model.eval()

        with torch.no_grad():
            for tst_step, data in enumerate(tqdm.tqdm(self.test_data_loader)):
                latent_dir = os.path.join(self.output_dir, "latents")
                if not os.path.exists(latent_dir):
                    os.makedirs(latent_dir)
                pickle_name = os.path.join(
                    latent_dir, "{}.lat".format(data["scene"][0])
                )
                # if os.path.exists(pickle_name):
                #     continue

                data = self.test_data_loader.dataset.load_data_to_gpu(data, self.device)

                # all scenes in dataset size < 512*512*128
                tsdf = data["gt_tsdf"]
                assert tsdf.shape[0] == 1
                default_value = 1.0
                original_dim = tsdf.shape

                # round all tsdf to blocks of 128, smaller size my have bad effect to the encoding/decoding
                # model of patchmode: patchsize=32
                new_tsdf, origin = self.round_vol(tsdf, default_value, 128)
                assert new_tsdf.shape[-1] == 128

                code1, code2, code1_q, code2_q, occl1 = self.get_first_stage_code(
                    model, new_tsdf, 128
                )

                latent_dict = {
                    "orginal_tsdf_dim": original_dim,
                    "padded_tsdf_dim": list(new_tsdf.shape),
                    "padded_voxel_origin": origin,
                    "tsdf_origin": data["gt_origin"].cpu(),
                    "scene": data["scene"][0],
                    "tsdf_voxel_size": data["voxel_size"].item(),
                    "code1": code1.cpu(),
                    "code2": code2.cpu(),
                    "occl1": occl1.cpu(),
                    "code1_q": code1_q.cpu(),
                    "code2_q": code2_q.cpu(),
                }

                with open(pickle_name, "wb") as f:
                    pickle.dump(latent_dict, f)

                if self.save_mesh:
                    mesh_dir = os.path.join(self.output_dir, "occ_meshes")
                    if not os.path.exists(mesh_dir):
                        os.makedirs(mesh_dir)
                    mesh_name = os.path.join(
                        mesh_dir,
                        "data_chk_step{}_{}.ply".format(tst_step, data["scene"][0]),
                    )
                    verts, faces = occ2mesh(occl1[0], thres=0.9, voxel_size=0.16)
                    mesh2ply(verts, faces, mesh_name)

    def load_state(self, log_file):
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"file not exist:{log_file}")
        params = None
        try:
            params = torch.load(log_file, map_location=lambda storage, loc: storage)
        except Exception as e:
            with open(log_file, "rb") as f:
                params = pickle.load(f)
        if params is not None:
            if self.model is not None:
                self.model.load_model_paras(params)
            else:
                raise AssertionError("model does not exist.")
            logging.info(f"loaded model params:{log_file}")
            # todo: retrive all status including: optimizer epoch log folder...
            status = params["status"]
            self.epoch = status["epoch"]
            self.global_step = status["global_step"]
            # if 'opt_paras' in params:
            #     for opt, opt_paras in zip(self.optimizer, params['opt_paras']):
            #         opt.load_state_dict(opt_paras)
        else:
            raise AssertionError("Fail to load params for model.")

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

    @torch.no_grad()
    def get_first_stage_code(self, model, tsdf_vol, cube_size_=96, decode=False):
        batch_size = tsdf_vol.shape[0]
        stride = cube_size = cube_size_
        ncubes = [i // cube_size for i in tsdf_vol.shape[2:]]
        minibatch = self.unfold_to_cubes(tsdf_vol, cube_size, stride)
        minibatch_size = 1
        code1_list = []
        code2_list = []
        code1q_list = []
        code2q_list = []
        occl1_list = []

        chunk_num = minibatch.shape[0] // minibatch_size
        assert minibatch.shape[0] % minibatch_size == 0
        chunks = torch.chunk(minibatch, chunk_num)

        for piece in chunks:
            code1, code2, code1_q, code2_q, occl1 = model.get_code(piece)
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

        # dec, occl2 = model.decode(code1q, code2q)
        return code1, code2, code1_q, code2_q, occl1
