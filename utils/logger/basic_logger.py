import builtins
import os
import datetime
import logging
import pickle
import shutil
import torch
import subprocess
from model.model_base import ModelBase
from utils.config.Configuration import Configuration
from tensorboardX import SummaryWriter
import pandas as pd
import torch

# todo: add lock while used in multiprocessing...


class LogTracker:
    """
    record training numerical indicators.
    """

    def __init__(self, *keys, phase="train"):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return {
            "{}/{}".format(self.phase, k): v
            for k, v in dict(self._data.average).items()
        }


class BasicLogger:
    logger = None

    def __new__(cls, *args, **kwargs):
        if cls.logger is None:
            cls.logger = super(BasicLogger, cls).__new__(cls)
            cls.logger.__initialized = False
        return cls.logger

    def __init__(self, config):
        if self.__initialized:
            return
        if not isinstance(config, Configuration):
            raise TypeError("input must be the Configuration type!")
        config_dict = config.get_complete_config()
        if "logging" not in config_dict.keys():
            raise KeyError("Not config on logger has been found!")
        self._program_version = None
        self._monitor_dict = {}
        self._status_hook = None
        self.root_log_dir = config_dict["logging"]["path"]
        self.log_suffix = config_dict["logging"]["suffix"]
        self._ckpt_eph_interval = config_dict["logging"]["ckpt_eph_interval"]
        if not isinstance(self._ckpt_eph_interval, int):
            self._ckpt_eph_interval = 0

        date_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if self.log_suffix is None or len(self.log_suffix) == 0:
            self._cur_instance_root_log_dir = date_time_str
        else:
            self._cur_instance_root_log_dir = "-".join([date_time_str, self.log_suffix])

        # overwrite by logdir
        if "log_dir" in config_dict["logging"]:
            self.root_log_dir = ""
            self._cur_instance_root_log_dir = config_dict["logging"]["log_dir"]

        self.complete_instance_dir = os.path.join(
            self.root_log_dir, self._cur_instance_root_log_dir
        )

        self._tensor_board_log_dir = os.path.join(
            self.complete_instance_dir, "tensor_board"
        )
        self._data_log_dir = os.path.join(self.complete_instance_dir, "data_log")
        self._model_para_log_dir = os.path.join(
            self.complete_instance_dir, "model_paras_log"
        )

        if not os.path.exists(self.complete_instance_dir):
            os.makedirs(self.complete_instance_dir)
        if not os.path.exists(self._tensor_board_log_dir):
            os.makedirs(self._tensor_board_log_dir)
        if not os.path.exists(self._data_log_dir):
            os.makedirs(self._data_log_dir)
        if not os.path.exists(self._model_para_log_dir):
            os.makedirs(self._model_para_log_dir)

        # add version file
        version_file = os.path.join(self.complete_instance_dir, "version.txt")
        self.get_program_version()
        self.log_version_info(version_file)
        self._tensor_board_writer = SummaryWriter(self._tensor_board_log_dir)
        self._tensor_board_writer_lf = None
        self._data_pickle_file = os.path.join(self._data_log_dir, "data_bin.pkl")
        self.__initialized = True

    def get_program_version(self):
        git_version = None
        try:
            git_version = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            )
            if self._program_version is None:
                self._program_version = git_version
        except:
            pass
        return git_version

    def log_version_info(self, file_name):
        with open(file_name, "w") as f:
            f.write(f"Current version:{self._program_version}")

    def log_config(self, config):
        if not isinstance(config, Configuration):
            raise TypeError("Please input a valid Configuration instance or reference")
        config.pack_configurations(self.complete_instance_dir)

    def log_data(
        self, data_name, data_content, add_to_tensorboard=False, step_key="global_step"
    ):
        status = self._status_hook()
        if isinstance(data_content, builtins.float):
            self._log_scalar(
                status, data_name, data_content, add_to_tensorboard, step_key
            )
        elif isinstance(data_content, builtins.int):
            self._log_scalar(
                status, data_name, data_content, add_to_tensorboard, step_key
            )
        else:
            raise NotImplementedError

    def log_mesh(self, data_name, verts, faces):
        if self._tensor_board_writer_lf is None:
            new_p = os.path.join(self._tensor_board_log_dir, "large_file_log")
            os.makedirs(new_p)
            self._tensor_board_writer_lf = SummaryWriter(new_p)
        self._tensor_board_writer_lf.add_mesh(
            data_name,
            vertices=verts,
            faces=faces,
            global_step=self._status_hook()["global_step"],
        )

    def log_image(self, data_name, image):
        if self._tensor_board_writer_lf is None:
            new_p = os.path.join(self._tensor_board_log_dir, "large_file_log")
            os.makedirs(new_p)
            self._tensor_board_writer_lf = SummaryWriter(new_p)
        self._tensor_board_writer_lf.add_image(
            data_name, image, global_step=self._status_hook()["global_step"]
        )

    def _add_to_pickle(self, status, data_name, data_content):
        with open(self._data_pickle_file, "ab") as f:
            pickle.dump(
                {"status": status, "name": data_name, "content": data_content}, f
            )

    def log_to_pickle(self, data_name, data_content):
        status = self._status_hook()
        self._add_to_pickle(status, data_name, data_content)

    def save_binary(self, sub_dir, file_name, data_content):
        sub_dir = os.path.join(self._data_log_dir, sub_dir)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        file_name = os.path.join(sub_dir, file_name)
        with open(file_name, "wb") as f:
            pickle.dump(data_content, f)

    def get_log_dir(self):
        return self._data_log_dir

    def _log_scalar(
        self,
        status,
        data_name,
        data_content,
        add_to_tensorboard=False,
        step_key="global_step",
    ):
        if data_name not in self._monitor_dict.keys():
            self._monitor_dict[data_name] = []
        self._monitor_dict[data_name].append((status, data_content))
        if add_to_tensorboard:
            self._tensor_board_writer.add_scalar(
                data_name, data_content, global_step=self._status_hook()[step_key]
            )
        self._add_to_pickle(status, data_name, data_content)

    def log_model_params(self, model, optimizers=None, force=False, suffix=""):
        # skip when ckpt_eph_interval is set
        status = self._status_hook()
        epoch = status["epoch"]
        global_step = status["global_step"]

        if (not force) and (epoch % self._ckpt_eph_interval != 0):
            return False
        if model is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module

        if not isinstance(model, ModelBase):
            raise TypeError("input type must have class attribute of ModelBase!")
        para_dict = {"status": status, "model_paras": model.state_dict()}
        if optimizers is not None:
            if not isinstance(optimizers, list):
                # assert isinstance(optimizers, torch.Optimizer)
                optimizers = [optimizers]
            opt_state_dict_list = []
            for i in optimizers:
                opt_state_dict_list += [i.state_dict()]
            para_dict.update({"opt_paras": opt_state_dict_list})
        date_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        pickle_name = (
            "-".join(
                [
                    f"model_ckpt-epoth{epoch}-globalstep{global_step}{suffix}",
                    date_time_str,
                ]
            )
            + ".pt"
        )
        with open(os.path.join(self._model_para_log_dir, pickle_name), "wb") as f:
            # pickle.dump(para_dict, f)
            torch.save(para_dict, f)
        logging.info(f"Log model state dict as: {pickle_name}")
        return True

    def register_status_hook(self, fn):
        if not callable(fn):
            raise TypeError(f"input must be a function!")
        self._status_hook = fn

    @classmethod
    def get_logger(cls, config=None):
        if cls.logger is not None:
            if config is not None:
                logging.warning("input config for logger will be ignored")
            return cls.logger
        if config is None:
            raise ValueError("config must be set")
        else:
            cls.logger = BasicLogger(config)
            return cls.logger

    def copy_screen_log(self, file_path):
        try:
            shutil.copy(
                file_path,
                os.path.join(self.complete_instance_dir, os.path.basename(file_path)),
            )
        except:
            logging.error("Fail to copy screen log...")
