from trainer.trainer_base import TrainerBase
import torch
import logging
from utils.logger.basic_logger import LogTracker
import os
import pickle


class SKetchVAETrainer(TrainerBase):
    def __init__(self, config):
        super().__init__()
        assert config.config_type in ["training", "testing"]
        if config.config_type == "training":
            self.optimizer_config = config["optimizer"]
            self.max_epoch = config["epoch"]
            self.enable_val = config.enable_val
            self.val_interval = config.val_interval
            self.train_metrics = LogTracker("total_loss", phase="train")
            self.train_log_dir = None
        elif config.config_type == "testing":
            self.test_config = config
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
        self.dataset.set_level("first")

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
                self.data_loader.dataset.load_data_to_gpu(data, self.device)

                self.optimizer.zero_grad()

                sketch = data["bev_sketch"].squeeze(-1)
                dec, posterior = model(sketch)
                loss = model.get_loss(sketch, dec, posterior)
                total_loss = loss["total"]
                total_loss.backward()

                self.optimizer.step()

                # print current status and logging
                if not self.distributed or self.rank == 0:
                    logging.info(
                        f"[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t"
                        f"global_step={self.global_step}\t"
                        f"loss={total_loss:.6f}\t"
                        #  f'tsdf_l1={tsdf_l1:.6f}\t'
                    )
                    self.logger.log_data("loss", total_loss.item(), True)
                    if step == 0:
                        self.logger.log_image("gt", sketch[0])
                        self.logger.log_image("rec", dec[0])

                self.step = step
                self.global_step += 1

            if not self.distributed or self.rank == 0:
                self.logger.log_model_params(self.model, optimizers=self.optimizer)

    def sum_val_dict(self, d):
        summary = {}
        for i in d:
            for k in i:
                if k not in summary:
                    summary[k] = []
                summary[k] += [i[k]]
        return summary

    def val_step(self):
        raise NotImplementedError

    def run_test(self):
        raise NotImplementedError

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
