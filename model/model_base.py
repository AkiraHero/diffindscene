import logging
import pickle

import torch
import torch.nn as nn


class ModelBase(nn.Module):
    module_class_list = {}

    def __init__(self):
        super(ModelBase, self).__init__()
        self.output_shape = None
        self.device = None
        self.mod_dict = nn.ModuleDict()

    def get_output_shape(self):
        if not self.output_shape:
            raise NotImplementedError
        return self.output_shape

    def set_device(self, device):
        self.device = device
        for k, v in self._modules.items():
            if "set_device" in v.__dir__():
                v.set_device(device)
        self.to(self.device)

    def set_eval(self):
        self.eval()
        for k, v in self.mod_dict.items():
            if "eval" in v.__dir__():
                v.eval()

    def set_attr(self, attr, value):
        pass

    def load_model_paras(self, params):
        if params is not None:
            super(ModelBase, self).load_state_dict(params["model_paras"])
        else:
            raise AssertionError("Fail to load params for model.")

    def load_model_paras_from_file(self, para_file):
        params = None
        try:
            params = torch.load(para_file, map_location=lambda storage, loc: storage)
        except Exception as e:
            with open(para_file, "rb") as f:
                params = pickle.load(f)
        self.load_model_paras(params)
        logging.info(f"loaded model params:{para_file}")

    def load_state_dict(self, file):
        raise AssertionError(
            "The load_state_dict function has been forbidden in this model system. "
            "Please use load_model_paras instead."
        )

    @staticmethod
    def check_config(config):
        required_paras = ["name", "paras"]
        ModelBase.check_config_dict(required_paras, config)

    @staticmethod
    def check_config_dict(required, config):
        assert isinstance(config, dict)
        for i in required:
            if i not in config.keys():
                err = f"Required config {i} does not exist."
                raise KeyError(err)
