import os
import argparse
import yaml
import shutil
import datetime
from easydict import EasyDict

"""
treat the configuration as a tree
"""


def default(config, attr, default_value):
    if attr in config:
        return config[attr]
    else:
        return default_value


# todo: control the access of members
class Configuration:
    def __init__(self):
        self.config_root_dir = None
        self.root_config = None
        self.expanded_config = None
        self.all_related_config_files = []
        self.dir_checked = False

        self._dataset_config = None
        self._training_config = None
        self._testing_config = None
        self._logging_config = None
        self._model_config = None
        self._extra_config = {"config_type": "extra"}

    @property
    def dataset_config(self):
        return EasyDict(self._dataset_config)

    @property
    def training_config(self):
        return EasyDict(self._training_config)

    @property
    def testing_config(self):
        return EasyDict(self._testing_config)

    @property
    def logging_config(self):
        return EasyDict(self._logging_config)

    @property
    def model_config(self):
        return EasyDict(self._model_config)

    @property
    def extra_config(self):
        return EasyDict(self._extra_config)

    def check_config_dir(self, config_dir):
        if not os.path.isdir(config_dir):
            raise IsADirectoryError(f"{config_dir} is not a valid directory.")
        # check subconfig dir
        if not os.path.isdir(os.path.join(config_dir, "dataset")):
            raise IsADirectoryError(f"{config_dir}/dataset is not a valid directory.")
        if not os.path.isdir(os.path.join(config_dir, "model")):
            raise IsADirectoryError(f"{config_dir}/model is not a valid directory.")
        if not os.path.isfile(os.path.join(config_dir, "root_config.yaml")):
            raise IsADirectoryError(
                f"{config_dir}/root_config.yaml is not a valid/existing file."
            )
        self.dir_checked = True

    def load_config(self, config_dir):
        self.config_root_dir = config_dir
        self.check_config_dir(self.config_root_dir)
        if self.dir_checked:
            self._load_root_config_file("root_config.yaml")

    def get_complete_config(self):
        if self.expanded_config is not None:
            return self.expanded_config.copy()
        raise TypeError("no complete config found!")

    def get_shell_args_train(self):
        parser = argparse.ArgumentParser(description="arg parser")
        parser.add_argument(
            "--cfg_dir",
            required=True,
            type=str,
            default=None,
            help="specify the config for training",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            required=False,
            help="batch size for training (in each process in distributed training)",
        )
        parser.add_argument(
            "--epoch",
            type=int,
            default=None,
            required=False,
            help="number of epochs to train for",
        )
        parser.add_argument(
            "--distributed",
            action="store_true",
            default=False,
            help="using distributed training",
        )
        parser.add_argument(
            "--local_rank",
            type=int,
            default=0,
            help="local rank for distributed training",
        )
        parser.add_argument(
            "--tcp_port",
            type=int,
            default=18888,
            help="tcp port for distributed training",
        )
        parser.add_argument(
            "--launcher",
            choices=["none", "pytorch", "slurm"],
            default="none",
            help="select distributed training launcher",
        )
        parser.add_argument(
            "--screen_log",
            type=str,
            default="scree_log",
            required=False,
            help="the file shell redirects to",
        )
        parser.add_argument(
            "--log_dir", required=False, type=str, default=None, help="log dir"
        )
        parser.add_argument(
            "--check_point_file",
            type=str,
            default=None,
            help="model checkpoint for pre-loading before training",
        )
        args = parser.parse_args()
        return args

    def get_shell_args_test(self):
        parser = argparse.ArgumentParser(description="arg parser")
        parser.add_argument(
            "--cfg_dir",
            required=True,
            type=str,
            default=None,
            help="specify the config for training",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            required=False,
            help="batch size for training (in each process in distributed training)",
        )
        parser.add_argument(
            "--distributed",
            action="store_true",
            default=False,
            help="using distributed testing",
        )
        parser.add_argument(
            "--local_rank",
            type=int,
            default=0,
            help="local rank for distributed testing",
        )
        parser.add_argument(
            "--tcp_port",
            type=int,
            default=18888,
            help="tcp port for distributed testing",
        )
        parser.add_argument(
            "--launcher",
            choices=["none", "pytorch", "slurm"],
            default="none",
            help="select distributed testing launcher",
        )
        parser.add_argument(
            "--screen_log", type=str, default=None, help="the file shell redirects to"
        )
        parser.add_argument(
            "--check_point_file",
            type=str,
            default=None,
            help="model checkpoint for pre-loading before testing",
        )
        args = parser.parse_args()
        return args

    def _load_yaml(self, file):
        abs_path = os.path.join(self.config_root_dir, file)
        with open(abs_path, "r") as f:
            return yaml.safe_load(f)

    def _load_root_config_file(self, config_file):
        self.root_config = self._load_yaml(config_file)
        self.expanded_config = self.root_config.copy()
        self.all_related_config_files.append(config_file)
        self._expand_config(self.expanded_config)
        # set corresponding config
        if "model" in self.expanded_config.keys():
            self._model_config = self.expanded_config["model"]
            self._model_config.update({"config_type": "model"})
        if "dataset" in self.expanded_config.keys():
            self._dataset_config = self.expanded_config["dataset"]
            self._dataset_config.update({"config_type": "dataset"})
        if "training" in self.expanded_config.keys():
            self._training_config = self.expanded_config["training"]
            self._training_config.update({"config_type": "training"})
        if "testing" in self.expanded_config.keys():
            self._testing_config = self.expanded_config["testing"]
            self._testing_config.update({"config_type": "testing"})
        if "logging" in self.expanded_config.keys():
            self._logging_config = self.expanded_config["logging"]
            self._logging_config.update({"config_type": "logging"})

    def _expand_config(self, config_dict):
        if not self._expand_cur_config(config_dict):
            if isinstance(config_dict, dict):
                for i in config_dict.keys():
                    sub_config = config_dict[i]
                    self._expand_config(sub_config)

    def _expand_cur_config(self, config_dict):
        if not isinstance(config_dict, dict):
            return False
        if "config_file" in config_dict.keys() and isinstance(
            config_dict["config_file"], str
        ):
            file_name = config_dict["config_file"]
            expanded = self._load_yaml(file_name)
            self._expand_config(expanded)
            self.all_related_config_files.append(file_name)
            config_dict["config_file"] = {"file_name": file_name, "expanded": expanded}
            return True
        return False

    def pack_configurations(self, _path):
        # all config file should be located in utils/config?? no
        # todo: pack config using expanded config
        shutil.copytree(self.config_root_dir, os.path.join(_path, "config"))

    @staticmethod
    def find_dict_node(target_dict, node_name):
        if not isinstance(target_dict, dict):
            raise TypeError
        res_parents = []
        res = Configuration._find_node_subtree(target_dict, node_name, res_parents)

        def flat_parents_list(parents, output):
            if len(parents) > 1:
                output.append(parents[0])
            else:
                return
            flat_parents_list(parents[1], output)

        output_parents = []
        flat_parents_list(res_parents, output_parents)
        return res, output_parents

    def find_node(self, node_name):
        return Configuration.find_dict_node(self.expanded_config, node_name)

    @staticmethod
    def _find_node_subtree(cur_node, keyword, parents_log=None):
        if isinstance(parents_log, list):
            parents_log.append(keyword)
        if not isinstance(cur_node, dict):
            return None
        res = Configuration._find_node_cur(cur_node, keyword)
        if res is None:
            for i in cur_node.keys():
                parents_log.clear()
                if isinstance(parents_log, list):
                    parents_log.append(i)
                new_parents_log = []
                parents_log.append(new_parents_log)
                res = Configuration._find_node_subtree(
                    cur_node[i], keyword, new_parents_log
                )
                if res is not None:
                    return res
        return res

    @staticmethod
    def _find_node_cur(cur_node, keyword):
        if not isinstance(cur_node, dict):
            return None
        for i in cur_node.keys():
            if i == keyword:
                return cur_node[i]
        return None

    def overwrite_value_by_keywords(
        self, parents_keywords_list, cur_keywords, new_value
    ):
        if not isinstance(self.expanded_config, dict):
            raise TypeError
        sub_dict_ref = self.expanded_config
        for key in parents_keywords_list:
            sub_dict_ref = sub_dict_ref[key]
        sub_dict_ref[cur_keywords] = new_value

    # only overwrite the first-found one on condition of equal keys
    def overwrite_config_by_shell_args(self, args):
        for name, value in args._get_kwargs():
            if value is not None:
                node, parents = self.find_node(name)
                if node is not None:
                    self.overwrite_value_by_keywords(parents, name, value)
                else:
                    self._extra_config[name] = value
