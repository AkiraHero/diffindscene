from trainer.trainer_base import TrainerBase
import torch
import logging
import tqdm
import torch.nn as nn
from utils.logger.basic_logger import LogTracker
from utils.diffusion_monitor import DiffusionMonitor
import os
import pickle
from utils.graphics_utils import save_tsdf_as_mesh
import numpy as np
from utils.visualize_occ import occ2mesh, mesh2ply
import time
import random
import cv2 as cv


class CascadedLDMTrainer(TrainerBase):
    def __init__(self, config):
        super().__init__()
        assert config.config_type in ["training", "testing"]
        if config.config_type == "training":
            self.optimizer_config = config["optimizer"]
            self.max_epoch = config["epoch"]
            self.enable_val = config.enable_val
            self.val_interval = config.val_interval
            self.diffusion_monitor = DiffusionMonitor()
            self.train_metrics = LogTracker("total_loss", phase="train")
            self.train_log_dir = None
        elif config.config_type == "testing":
            self.test_config = config
            self.save_mesh = config.save_mesh
            self.test_log_dir = config.test_log_dir
            if not os.path.exists(self.test_log_dir):
                os.makedirs(self.test_log_dir)

        if not self.distributed:
            self.device = torch.device(config["device"])

    def set_optimizer(self, optimizer_config):
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[0]["type"]]
        self.optimizer = optimizer_ref(
            model.get_trainable_parameters(), **optimizer_config[0]["paras"]
        )
        logging.info("[Optimizer Paras]" + str(optimizer_config[0]["paras"]))

    def run(self):
        # torch.autograd.set_detect_anomaly(True)
        if not self.check_ready():
            raise ModuleNotFoundError(
                "The trainer not ready. Plz set model/dataset first"
            )
        super().run()
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        self.dataset.set_level(model.model_level)
        # self.global_step = 0
        if not self.distributed or self.rank == 0:
            self.diffusion_monitor.set_scheduler(model.scheduler)

        for epoch in range(self.max_epoch):
            if self.enable_val:
                if not self.distributed or self.rank == 0:
                    if self.train_log_dir is None:
                        self.train_log_dir = self.logger.get_log_dir()
                    if epoch > 0 and epoch % self.val_interval == 0:
                        self.is_val = True
                        logging.info(
                            "\n\n\n------------------------------Validation Start------------------------------"
                        )
                        self.val_step()

            self.is_val = False
            self.epoch = epoch
            model.train()
            self.train_metrics.reset()
            self.data_loader.dataset.epoch = epoch
            for step, data in enumerate(self.data_loader):
                data = self.data_loader.dataset.load_data_to_gpu(data, self.device)

                self.optimizer.zero_grad()
                # if not self.distributed or self.rank == 0:
                if self.global_step == 0:
                    model.prepare_training(data)

                # use DistributedDataParallel for torch dist
                loss = self.model(data)

                total_loss = loss["total"]
                total_loss.backward()

                self.optimizer.step()
                if not self.distributed or self.rank == 0:
                    loss.pop("total")
                    self.diffusion_monitor.update_loss(loss)

                # for ema
                model.on_train_batch_end()

                # print current status and logging
                if not self.distributed or self.rank == 0:
                    logging.info(
                        f"[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t"
                        f"global_step={self.global_step}\t"
                        f"loss={total_loss:.6f}\t"
                        #  f'tsdf_l1={tsdf_l1:.6f}\t'
                    )
                    self.logger.log_data("loss", total_loss.item(), True)

                    self.train_metrics.update("total_loss", total_loss.item())
                    train_metrics_epc = self.train_metrics.result()
                    for key, value in train_metrics_epc.items():
                        logging.info("{:5s}: {}\t".format(str(key), value))
                        self.logger.log_data(str(key) + "_aver_cur_epc", value, True)

                self.step = step
                self.global_step += 1

            if not self.distributed or self.rank == 0:
                self.logger.log_model_params(self.model, optimizers=self.optimizer)
                gamma_intertal, loss_aver = self.diffusion_monitor.get_gamma_loss_dist()
                for seg, aver in zip(gamma_intertal, loss_aver):
                    data_label = "gamma_loss/{:.3f}-{:.3f}".format(seg[0], seg[1])
                    self.logger.log_data(data_label, aver.cpu().item(), True)

    @staticmethod
    def L1metric(input, target):
        with torch.no_grad():
            loss = nn.L1Loss()
            output = loss(input, target)
        return output

    @staticmethod
    def visualize_out(outdata_dict, save_dir):
        # generated_samples = outdata_dict['output']
        # bs = generated_samples.shape[0]
        voxel_size = {"first": 0.16, "second": 0.04, "third": 0.04}
        for level in outdata_dict.keys():
            result = outdata_dict[level]["result"]
            if result is None:
                continue

            # save data
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            level_dict_file = os.path.join(save_dir, "{}.pickle".format(level))
            with open(level_dict_file, "wb") as f:
                pickle.dump(outdata_dict[level], f)

            val_num = result.shape[0]
            for i in range(val_num):
                result_item = outdata_dict[level]["result"][i]

                mesh_dir = os.path.join(save_dir)
                if not os.path.exists(mesh_dir):
                    os.makedirs(mesh_dir)

                if "time_consumption" in outdata_dict[level]["info"]:
                    description_file = os.path.join(
                        mesh_dir, "time_{}.txt".format(level)
                    )
                    with open(description_file, "w") as f:
                        f.write(
                            "{}".format(outdata_dict[level]["info"]["time_consumption"])
                        )

                if level in ["first", "second"]:
                    occ_item = outdata_dict[level]["next_occ"][i]
                    mesh_name = os.path.join(
                        mesh_dir, "gen_occ_level_{}.ply".format(level)
                    )
                    verts, faces = occ2mesh(
                        occ_item, thres=0.9, voxel_size=voxel_size[level]
                    )
                    mesh2ply(verts, faces, mesh_name)
                if level in ["first"]:
                    if "sketch" in outdata_dict[level]["info"]:
                        sketch_item = outdata_dict[level]["info"]["sketch"][i]
                        sketch_item = sketch_item.squeeze(0).squeeze(-1).cpu().numpy()
                        sketch_file_name = os.path.join(mesh_dir, "sketch.png")
                        cv.imwrite(sketch_file_name, sketch_item * 255)
                    if "text" in outdata_dict[level]["info"]:
                        description = outdata_dict[level]["info"]["text"][i]
                        description_file = os.path.join(mesh_dir, "text.txt")
                        with open(description_file, "w") as f:
                            f.write(str(description))
                    if "scene_size" in outdata_dict[level]["info"]:
                        description_file = os.path.join(mesh_dir, "scene_size.txt")
                        scene_size = outdata_dict[level]["info"]["scene_size"]
                        with open(description_file, "w") as f:
                            scn_size_str = "{},{},{}".format(
                                scene_size[0], scene_size[1], scene_size[2]
                            )
                            f.write(scn_size_str)

                if level in ["third"]:
                    mesh_name = os.path.join(
                        mesh_dir, "gen_tsdf_level_{}_0.ply".format(level)
                    )
                    save_tsdf_as_mesh(
                        result_item.cpu().numpy(),
                        (0, 0, 0),
                        voxel_size[level],
                        mesh_name,
                        level=0,
                    )

    def sum_val_dict(self, d):
        summary = {}
        for i in d:
            for k in i:
                if k not in summary:
                    summary[k] = []
                summary[k] += [i[k]]
        return summary

    def val_step(self):
        model_v = self.model
        model_v = model_v.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_v = self.model.module
        with torch.no_grad():
            # save ckpt
            self.logger.log_model_params(model_v, force=True, suffix="val_noema")
            with model_v.ema_scope():
                self.logger.log_model_params(model_v, force=True, suffix="val_ema")

    def run_test(self):
        # assert self.distributed == False
        super().run_test()
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        self.global_step = 0
        model.eval()
        self.test_data_loader.dataset.set_level(model.model_level)
        model.set_operating_size(self.test_config.operating_size)
        model.level_list = self.test_config.level_seq
        with torch.no_grad():
            for tst_step, data in enumerate(tqdm.tqdm(self.test_data_loader)):
                data = self.test_data_loader.dataset.load_data_to_gpu(data, self.device)

                seed = random.randint(0, 0xFFFF_FFFF_FFFF_FFFF)
                torch.manual_seed(seed)

                outdata_dict = model.restoration(data)
                log_dir = os.path.join(self.test_log_dir, str(tst_step))
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                with open(os.path.join(log_dir, "seed.txt"), "w") as f:
                    f.write("{}".format(seed))
                self.visualize_out(outdata_dict, log_dir)



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
