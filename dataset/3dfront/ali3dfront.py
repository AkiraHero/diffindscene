import os
import numpy as np
import pickle
import logging
import torch

from torch.utils.data import DataLoader
from collections import OrderedDict

from dataset.dataset_base import DatasetBase
from utils.config.Configuration import default

from .tsdf_transforms import ToTensor, RandomTransformSpace, Compose
from .latent_transforms import RandomTransform, SimpleCrop


class Ali3DFront(DatasetBase):
    support_data_contents = ["tsdf", "latent", "sketch"]

    def __init__(self, config):
        super(Ali3DFront, self).__init__()
        self._batch_size = config.paras.batch_size
        self._shuffle = config.paras.shuffle
        self._num_workers = config.paras.num_workers
        self.data_root = config.paras.data_root
        self.data_split_file = config.paras.data_split_file

        self.version = default(config.paras, "version", "")
        self.load_content = default(config.paras, "load_content", ["tsdf"])
        self.latent_dir = default(config.paras, "latent_dir", None)
        self.latent_scale = default(config.paras, "latent_scale", None)
        self.level_config = default(config.paras, "level_config", None)
        self.voxel_dim = default(config.paras, "voxel_dim", [256, 256, 256])

        self.batch_collate_function_name = default(
            config.paras, "batch_collate_func", "batch_collate_fn"
        )
        self.designated_batch_collate_function = self.__getattribute__(
            self.batch_collate_function_name
        )

        self.gen_sketch = False
        self.config_load_content()

        self.tsdf_cashe = OrderedDict()
        self.max_cashe = 1000

        self.transform_list = default(config.paras, "transform", [])
        self.set_transform()

        self.mode = "train"
        if "mode" in config.paras:
            self.set_mode(config.paras.mode)

        self.mode_indices = {"train": [], "test": [], "val": []}
        self.scene_id_list, self.scene_path_dict = self.load_scene_id_list(
            self.data_root
        )
        self.get_train_val_split()

    def config_load_content(self):
        for i in self.load_content:
            assert i in self.support_data_contents
        if "latent" in self.load_content:
            latent_files = os.listdir(self.latent_dir)
            self.latent_file_paths = {}
            for i in latent_files:
                file_path = os.path.join(self.latent_dir, i)
                scene_id = i.split(".")[0]
                self.latent_file_paths[scene_id] = file_path
        if "sketch" in self.load_content:
            self.gen_sketch = True

    def set_transform(self):
        if hasattr(self, "transforms"):
            del self.transforms
        # set transform for data producing/augmentation
        transform = []
        # highest resolution
        voxel_dim = self.voxel_dim
        voxel_size = 0.04  # max voxel

        random_rotation = True
        random_translation = True
        if "train" != self.mode:
            random_rotation = False
            random_translation = False
        paddingXY = 0.12
        paddingZ = 0.12
        epochs = 999

        transform += [
            ToTensor(),
        ]

        logging.info("[Dataset]transform list:" + str(self.transform_list))
        if len(self.transform_list) == 0:
            logging.warning("[Dataset]No transform added for dataset!!")

        elif self.transform_list[0] == "randomcrop":
            transform += [
                RandomTransformSpace(
                    voxel_dim,
                    voxel_size,
                    random_rotation,
                    random_translation,
                    paddingXY,
                    paddingZ,
                    max_epoch=epochs,
                    using_camera_pose=False,
                    random_trans_method="occ_center",
                    random_rot_method="right-angle",
                ),
            ]
        elif self.transform_list[0] == "simpletrans":
            transform += [
                RandomTransform(
                    voxel_dim,
                    voxel_size,
                    random_rotation,
                    random_translation,
                    paddingXY,
                    paddingZ,
                    max_epoch=epochs,
                    gen_bev_sketch=self.gen_sketch,
                )
            ]
        else:
            raise NotImplementedError

        if "simplecrop" in self.transform_list:
            transform += [SimpleCrop(voxel_dim)]

        self.transforms = Compose(transform)

    def set_level(self, level):
        logging.info(
            "[DataSet]Ali3DFront changed model level related config:{}".format(level)
        )
        assert level in ["first", "second", "third"]
        assert self.level_config is not None
        level_config = self.level_config[level]
        self.load_content = level_config["load_content"]
        self.transform_list = level_config["transform"]
        self.config_load_content()
        self.set_transform()

    def read_scene_list(self, file):
        with open(file, "r") as f:
            scenes = f.readlines()
            scenes = [i.strip("\n") for i in scenes]
            return scenes

    def get_train_val_split(self):
        cur_dir = os.path.dirname(__file__)
        train_list = self.read_scene_list(
            os.path.join(cur_dir, self.data_split_file["train"])
        )

        # no need for generative model
        test_list = []
        val_list = []

        scene_mode_dict = {}
        for i in train_list:
            scene_mode_dict[i] = "train"
        for i in val_list:
            scene_mode_dict[i] = "val"
        for i in test_list:
            scene_mode_dict[i] = "test"

        for inx, i in enumerate(self.scene_id_list):
            scene = i
            if scene in self.scene_path_dict and scene in scene_mode_dict:
                mode = scene_mode_dict[scene]
                self.mode_indices[mode].append(inx)

    def set_mode(self, mode):
        self.tsdf_cashe.clear()
        assert mode in ["train", "val", "test"]
        self.mode = mode

    def map_idx(self, idx):
        mode_str = self.get_mode_indices_label()
        return self.mode_indices[mode_str][idx]

    def get_mode_indices_label(self):
        mode_str = self.mode
        if self.mode == "test_scene":
            mode_str = "test"
        return mode_str

    def load_scene_id_list(self, data_dir):
        scene_id_list = [
            i for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))
        ]
        # scene path
        scene_path_list = {}
        for scene in scene_id_list:
            p = os.path.join(data_dir, scene)
            scene_path_list[scene] = p
        return scene_id_list, scene_path_list

    def read_latent(self, scene):
        lat_file = self.latent_file_paths[scene]
        with open(lat_file, "rb") as f:
            lat_dict = pickle.load(f)
        return lat_dict

    def read_scene_volumes(self, scene):
        def switch_dim(full_tsdf_dict):
            # switch dims to x,y,z
            tsdf = full_tsdf_dict["tsdf"]
            tsdf = tsdf.transpose(0, 2, 1)
            origin = np.array(
                [
                    full_tsdf_dict["origin"][0],
                    full_tsdf_dict["origin"][2],
                    full_tsdf_dict["origin"][1],
                ]
            )
            voxel_size = full_tsdf_dict["voxel_size"]
            return {"tsdf": tsdf, "origin": origin, "voxel_size": voxel_size}

        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe.popitem(last=False)
            scene_path = self.scene_path_dict[scene]

            self.tsdf_cashe[scene] = {}

            if self.version == "v2":
                filename = "tsdf_v2.npz"
            else:
                filename = "tsdf.npz"
            full_tsdf_dict = np.load(os.path.join(scene_path, filename))
            full_tsdf_dict = switch_dim(full_tsdf_dict)
            self.tsdf_cashe[scene]["gt"] = full_tsdf_dict["tsdf"]
            self.tsdf_cashe[scene]["gt_origin"] = full_tsdf_dict["origin"]
            self.tsdf_cashe[scene]["voxel_size"] = full_tsdf_dict["voxel_size"]
            self.tsdf_cashe[scene]["crop_dim"] = torch.tensor(self.voxel_dim)
            self.tsdf_cashe[scene]["data_type"] = "tsdf"
        return self.tsdf_cashe[scene]

    def __len__(self):
        return len(self.mode_indices[self.get_mode_indices_label()])

    def __getitem__(self, idx):
        return self.get_general_item(idx)

    def get_general_item(self, idx):
        idx = self.map_idx(idx)
        scene = self.scene_id_list[idx]

        items = {
            "scene": scene,
        }

        if "tsdf" in self.load_content:
            tsdf = self.read_scene_volumes(scene)
            items.update({"tsdf": tsdf, "vol_type": tsdf["data_type"]})

        if "latent" in self.load_content:
            latent = self.read_latent(scene)
            items.update(
                {"latent": latent, "latent_scale": torch.tensor(self.latent_scale)}
            )
        try:
            if self.transforms is not None:
                items = self.transforms(items)
        except:
            logging.error("[Dataset] Transform failure on scene:{}".format(scene))
            raise RuntimeError
        return items

    def get_data_loader(self, distributed=False):
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(self)
        else:
            sampler = None

        bfn = self.designated_batch_collate_function

        data_loader = DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=(sampler is None) and self._shuffle,
            num_workers=self._num_workers,
            collate_fn=bfn,
            pin_memory=True,
            drop_last=False,
            sampler=sampler,
        )
        return data_loader

    @staticmethod
    def batch_collate_fn_for_lat(batch_list, _unused=False):
        assert len(batch_list) == 1  # for they hv diff size
        batch_dict = {}
        batch_dict["scene"] = []
        batch_dict["gt_tsdf"] = []
        batch_dict["gt_origin"] = []
        batch_dict["voxel_size"] = []
        for i in batch_list:
            batch_dict["scene"].append(i["scene"])
            batch_dict["gt_tsdf"].append(i["tsdf"]["gt"].unsqueeze(0))
            batch_dict["gt_origin"].append(i["tsdf"]["gt_origin"])
            batch_dict["voxel_size"].append(i["tsdf"]["voxel_size"])
        for k in ["gt_tsdf", "gt_origin", "voxel_size"]:
            batch_dict[k] = torch.stack(batch_dict[k])
        return batch_dict

    @staticmethod
    def batch_collate_latent_code(batch_list, _unused=False):
        batch_dict = {}
        keys = batch_list[0].keys()
        keep_list_key = ["scene", "latent_scale"]
        for k in keys:
            batch_dict[k] = []
        for i in batch_dict:
            for j in batch_list:
                batch_dict[i].append(j[i])
        for k in batch_dict:
            if k not in keep_list_key:
                batch_dict[k] = torch.cat(batch_dict[k])
        return batch_dict

    @staticmethod
    def batch_collate_fn(batch_list, _unused=False):
        batch_dict = {}
        batch_dict["scene"] = []
        batch_dict["gt_tsdf"] = []
        batch_dict["vol_origin_partial"] = []
        batch_dict["voxel_size"] = batch_list[0]["tsdf"]["voxel_size"]

        for i in batch_list:
            batch_dict["scene"].append(i["scene"])
            batch_dict["gt_tsdf"].append(i["partial_tsdf"]["gt"])
            batch_dict["vol_origin_partial"].append(i["vol_origin_partial"])

        batch_dict["gt_tsdf"] = torch.stack(batch_dict["gt_tsdf"]).unsqueeze(
            1
        )  # chn = 1
        batch_dict["vol_type"] = batch_list[0]["vol_type"]
        batch_dict["vol_origin_partial"] = torch.stack(batch_dict["vol_origin_partial"])
        return batch_dict

    @staticmethod
    def load_data_to_gpu(batch_dict, device=None):
        batch_dict_gpu = {}
        for i in batch_dict:
            if isinstance(batch_dict[i], torch.Tensor):
                batch_dict_gpu[i] = batch_dict[i].to(device)
            else:
                batch_dict_gpu[i] = batch_dict[i]
        return batch_dict_gpu
