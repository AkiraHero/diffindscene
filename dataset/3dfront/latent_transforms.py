from typing import Any
import numpy as np
import torch
from functools import partial
import cv2 as cv


def get_random_sketch(tsdf, resize=(128, 128), use_random=False):
    def mapping_function(x, y, freq_x=0.01, freq_y=0.01, scale_x=10, scale_y=10):
        new_x = x + np.sin(freq_x * y + 10) * scale_x
        new_y = y + np.sin(freq_y * x + 10) * scale_y
        return new_x, new_y

    z_dim_len = tsdf.shape[1]
    occ_coord = ((tsdf.abs()) < 1.0).nonzero()
    max_occ_coord, _ = occ_coord.max(dim=0)
    min_occ_coord, _ = occ_coord.min(dim=0)
    z_min = min_occ_coord[2]
    z_max = max_occ_coord[2]
    # find occ
    mid_z_min = int(z_min + (z_max - z_min) / 4)
    mid_z_max = int(z_min + (z_max - z_min) / 4 * 3)
    mid_z = torch.randint(mid_z_min, mid_z_max, ())
    mid_slice = tsdf[:, :, mid_z]

    mid_occ = (mid_slice.abs()) < 1.0
    sketch = np.zeros_like(mid_occ, dtype=np.float32)
    sketch[mid_occ] = 1.0

    # padding the sketch
    if use_random:
        pad_width = int(10 + np.random.rand() * 30)
        sketch = np.pad(sketch, pad_width=pad_width, mode="constant", constant_values=0)
        height, width = sketch.shape[:2]

        fr_x, fr_y, scale_x, scale_y = (
            np.random.rand() * 0.02,
            np.random.rand() * 0.02,
            np.random.rand() * 20,
            np.random.rand() * 20,
        )
        remap_map = np.zeros((height, width, 2), np.float32)

        x_coords, y_coords = np.meshgrid(range(width), range(height))
        coords = np.stack((x_coords.flatten(), y_coords.flatten()), axis=1)
        x, y = coords[:, 0], coords[:, 1]
        new_x, new_y = mapping_function(x, y, fr_x, fr_y, scale_x, scale_y)
        remap_map[y, x] = np.stack([new_x, new_y]).transpose()

        output_image = cv.remap(sketch, remap_map, None, cv.INTER_LINEAR)
    else:
        output_image = sketch
    if resize is not None:
        output_image = cv.resize(output_image, resize)
    return output_image


class RandomTransform(object):
    """Apply a random 3x4 linear transform to the world coordinate system.
    This affects pose as well as TSDFs.
    """

    def __init__(
        self,
        voxel_dim,
        voxel_size,
        random_rotation=True,
        random_translation=True,
        paddingXY=1.5,
        paddingZ=0.25,
        max_epoch=999,
        max_depth=3.0,
        gen_bev_sketch=False,
    ):
        """
        Args:
            voxel_dim: tuple of 3 ints (nx,ny,nz) specifying
                the size of the output volume
            voxel_size: floats specifying the size of a voxel
            random_rotation: wheater or not to apply a random rotation
            random_translation: wheater or not to apply a random translation
            paddingXY: amount to allow croping beyond maximum extent of TSDF
            paddingZ: amount to allow croping beyond maximum extent of TSDF
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            max_epoch: maximum epoch
            max_depth: maximum depth
        """

        self.voxel_dim = voxel_dim
        # self.origin = origin
        self.voxel_size = voxel_size
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.max_depth = max_depth
        self.padding_start = torch.Tensor([paddingXY, paddingXY, paddingZ])
        # no need to pad above (bias towards floor in volume)
        self.padding_end = torch.Tensor([paddingXY, paddingXY, 0])
        self.gen_bev_sketch = gen_bev_sketch

    @staticmethod
    def transpose2(vol, dim_1, dim_2):
        vol_ = torch.flip(vol, dims=[dim_1, dim_2])
        vol_ = torch.transpose(vol_, dim_1, dim_2)
        return vol_

    def gen_trans_function(self, dim_1, dim_2):
        trans_funct = []
        for i in range(4):
            trans_funct += [
                partial(torch.rot90, k=i, dims=[dim_1, dim_2])
            ]  # dim0/1 x/y plane
        trans_funct += [partial(torch.flip, dims=[dim_1])]
        trans_funct += [partial(torch.flip, dims=[dim_2])]
        trans_funct += [partial(torch.transpose, dim0=dim_1, dim1=dim_2)]
        trans_funct += [partial(self.transpose2, dim_1=dim_1, dim_2=dim_2)]
        return trans_funct

    def get_random_vol(self, vol, dim1, dim2, fix_inx=None):
        trans_funct = self.gen_trans_function(dim1, dim2)
        # vol can be augmented by group of flip, transpose, rotation
        if fix_inx is None:
            trans_num = len(trans_funct)
            inx = torch.randint(0, trans_num, (1,))
        else:
            inx = fix_inx
        return trans_funct[inx](vol), inx

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

    def __call__(self, data):
        if "tsdf" in data and "latent" not in data:
            tsdf_vol = data["tsdf"]["gt"]
            tsdf_vol, _ = self.get_random_vol(tsdf_vol, dim1=0, dim2=1)
            A = tsdf_vol.shape[0]
            B = tsdf_vol.shape[1]
            C = tsdf_vol.shape[2]
            assert A <= self.voxel_dim[0]
            assert B <= self.voxel_dim[1]
            assert C <= self.voxel_dim[2]

            target_volume = torch.ones(*self.voxel_dim)
            start_x = (self.voxel_dim[0] - A) // 2
            start_y = (self.voxel_dim[1] - B) // 2
            start_z = (self.voxel_dim[2] - C) // 2
            target_volume[
                start_x : start_x + A, start_y : start_y + B, start_z : start_z + C
            ] = tsdf_vol

            data["partial_tsdf"] = {}
            data["partial_tsdf"]["gt_origin"] = torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float
            )
            data["partial_tsdf"]["gt"] = [target_volume]
            data["vol_origin_partial"] = torch.tensor([0.0, 0.0, 0.0])
            return data
        elif "latent" in data:
            # hard code:
            # bug z dim should be 16
            code1_padding_shape = [64, 64, 16]
            code2_padding_shape = [128, 128, 32]
            tsdf_padding_shape = [512, 512, 128]

            initial_tsdf_dim = data["latent"]["orginal_tsdf_dim"]
            if len(initial_tsdf_dim) == 5:
                initial_tsdf_dim = initial_tsdf_dim[2:]
            else:
                assert initial_tsdf_dim == 3

            # get minimum tsdf vol with factor 32, and corresponding code
            # padded_voxel_origin is general 0 or 32*n
            tsdf_dim_32 = [32 * ((i - 1) // 32 + 1) for i in initial_tsdf_dim]
            tsdf_code1_shape = [i // 8 for i in tsdf_dim_32]
            tsdf_code2_shape = [i // 4 for i in tsdf_dim_32]
            tsdf_code1_origin = [i // 8 for i in data["latent"]["padded_voxel_origin"]]
            tsdf_code2_origin = [i // 4 for i in data["latent"]["padded_voxel_origin"]]
            if "tsdf" in data:
                tsdf_vol = data["tsdf"]["gt"].unsqueeze(0).unsqueeze(0)
                if self.gen_bev_sketch:
                    bev_sketch = get_random_sketch(
                        tsdf_vol.squeeze(0).squeeze(0), resize=None
                    )
                    bev_sketch = (
                        torch.tensor(bev_sketch).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                    )

            code1 = data["latent"]["code1"]
            code1_mask = torch.zeros_like(code1)
            code1_mask[
                ...,
                tsdf_code1_origin[0] : tsdf_code1_origin[0] + tsdf_code1_shape[0],
                tsdf_code1_origin[1] : tsdf_code1_origin[1] + tsdf_code1_shape[1],
                tsdf_code1_origin[2] : tsdf_code1_origin[2] + tsdf_code1_shape[2],
            ] = 1

            occl1 = data["latent"]["occl1"]
            code2 = data["latent"]["code2"]
            code2_mask = torch.zeros_like(code2)
            code2_mask[
                ...,
                tsdf_code2_origin[0] : tsdf_code2_origin[0] + tsdf_code2_shape[0],
                tsdf_code2_origin[1] : tsdf_code2_origin[1] + tsdf_code2_shape[1],
                tsdf_code2_origin[2] : tsdf_code2_origin[2] + tsdf_code2_shape[2],
            ] = 1

            # extend mask to fixed shape
            padded_code1 = self.round_vol(code1, 0.0, code1_padding_shape)
            code1_mask = self.round_vol(code1_mask, 0.0, code1_padding_shape)

            padded_code2 = self.round_vol(code2, 0.0, code2_padding_shape)
            code2_mask = self.round_vol(code2_mask, 0.0, code2_padding_shape)
            # the occ score value before sigmoid
            padded_occl1 = self.round_vol(occl1, -1000.0, code2_padding_shape)
            if "tsdf" in data:
                # pad the tsdf to the shape which has been used to generate the latent code
                tsdf_origin = data["latent"]["padded_voxel_origin"]
                padded_tsdf_vol = self.round_vol(
                    tsdf_vol,
                    1.000,
                    data["latent"]["padded_tsdf_dim"][2:],
                    pad_origin=list(tsdf_origin),
                )
                # pad the tsdf according to the padding of codes
                tsdf_mask = torch.zeros_like(padded_tsdf_vol)
                tsdf_mask[
                    ...,
                    tsdf_origin[0] : tsdf_origin[0] + tsdf_dim_32[0],
                    tsdf_origin[1] : tsdf_origin[1] + tsdf_dim_32[1],
                    tsdf_origin[2] : tsdf_origin[2] + tsdf_dim_32[2],
                ] = 1

                padded_tsdf_vol = self.round_vol(
                    padded_tsdf_vol, 1.000, tsdf_padding_shape
                )
                tsdf_mask = self.round_vol(tsdf_mask, 0.0, tsdf_padding_shape)
                if self.gen_bev_sketch:
                    bev_sketch = self.round_vol(
                        bev_sketch,
                        0,
                        data["latent"]["padded_tsdf_dim"][2:4] + [1],
                        pad_origin=list(tsdf_origin),
                    )
                    bev_sketch = self.round_vol(
                        bev_sketch, 0, tsdf_padding_shape[0:2] + [1]
                    )

            # augmentation
            padded_code1, inx = self.get_random_vol(padded_code1, dim1=2, dim2=3)
            padded_code2, _ = self.get_random_vol(
                padded_code2, dim1=2, dim2=3, fix_inx=inx
            )
            code1_mask, _ = self.get_random_vol(code1_mask, dim1=2, dim2=3, fix_inx=inx)
            code2_mask, _ = self.get_random_vol(code2_mask, dim1=2, dim2=3, fix_inx=inx)
            padded_occl1, _ = self.get_random_vol(
                padded_occl1, dim1=2, dim2=3, fix_inx=inx
            )
            if "tsdf" in data:
                padded_tsdf_vol, _ = self.get_random_vol(
                    padded_tsdf_vol, dim1=2, dim2=3, fix_inx=inx
                )
                tsdf_mask, _ = self.get_random_vol(
                    tsdf_mask, dim1=2, dim2=3, fix_inx=inx
                )
                if self.gen_bev_sketch:
                    bev_sketch, _ = self.get_random_vol(
                        bev_sketch, dim1=2, dim2=3, fix_inx=inx
                    )

            data_dict = {}
            data_dict["padded_code1"] = padded_code1
            data_dict["padded_code2"] = padded_code2
            data_dict["code1_mask"] = code1_mask.to(torch.bool)
            data_dict["code2_mask"] = code2_mask.to(torch.bool)
            data_dict["padded_occl1"] = padded_occl1

            # shape has been changed..
            code1_min_coord, _ = code1_mask.nonzero().min(dim=0)
            code1_max_coord, _ = code1_mask.nonzero().max(dim=0)

            code2_min_coord, _ = code2_mask.nonzero().min(dim=0)
            code2_max_coord, _ = code2_mask.nonzero().max(dim=0)

            data_dict["code1_shape"] = (code1_max_coord - code1_min_coord + 1)[2:]
            data_dict["code2_shape"] = (code2_max_coord - code2_min_coord + 1)[2:]

            # the original shape of tsdf without any data augmentation
            data_dict["initial_tsdf_shape"] = torch.tensor(initial_tsdf_dim)
            if "tsdf" in data:
                data_dict["padded_tsdf"] = padded_tsdf_vol
                data_dict["tsdf_mask"] = tsdf_mask
                if self.gen_bev_sketch:
                    data_dict["bev_sketch"] = bev_sketch
            data_dict["scene"] = data["scene"]
            if "scene_description" in data:
                data_dict["scene_description"] = data["scene_description"]
            data_dict["latent_scale"] = data["latent_scale"]
            return data_dict

    @staticmethod
    def crop_vol():
        pass

    def __repr__(self):
        return self.__class__.__name__


class SimpleCrop(object):
    def __init__(self, crop_dim) -> None:
        self.voxel_dim = crop_dim

    def __call__(self, data) -> Any:
        data_dict = {}

        code1_shape = data["code1_shape"]
        # get croppable dim of these vols
        crop_code1_dim = torch.tensor(
            [min(i // 8, j) for i, j in zip(self.voxel_dim, code1_shape)]
        )
        crop_code2_dim = torch.tensor([i * 2 for i in crop_code1_dim])
        crop_tsdf_dim = torch.tensor([i * 4 for i in crop_code2_dim])

        # get origin range of start point of cropped code1, relative to [valid] code1 vol
        code1_st_point_min = [0, 0, 0]
        code1_st_point_max = [i - j for i, j in zip(code1_shape, crop_code1_dim)]

        # get code1 origin relative to [whole] code1 vol
        code1_origin = data["code1_mask"].nonzero().min(dim=0)[0][2:]

        # get cropped vols origin relative to initial [valid] vol
        code1_st_point = torch.tensor(
            [
                torch.randint(i, j + 1, (1,))
                for i, j in zip(code1_st_point_min, code1_st_point_max)
            ]
        )
        code2_st_point = code1_st_point * 2
        tsdf_st_point = code2_st_point * 4

        # get cropped vols origin relative to initial [whole] vol
        code1_st_point += code1_origin
        code1_ed_point = code1_st_point + crop_code1_dim

        # only modify mask
        code1_crop_mask = torch.full(data["code1_mask"].shape, False)
        code1_crop_mask[
            :,
            :,
            code1_st_point[0] : code1_ed_point[0],
            code1_st_point[1] : code1_ed_point[1],
            code1_st_point[2] : code1_ed_point[2],
        ] = True

        cropped_code1 = data["padded_code1"][
            :,
            :,
            code1_st_point[0] : code1_ed_point[0],
            code1_st_point[1] : code1_ed_point[1],
            code1_st_point[2] : code1_ed_point[2],
        ]

        # get code2 origin relative to [whole] code1 vol
        code2_origin = data["code2_mask"].nonzero().min(dim=0)[0][2:]
        # get cropped vols origin relative to initial [whole] vol
        code2_st_point += code2_origin
        code2_ed_point = code2_st_point + crop_code2_dim
        cropped_code2 = data["padded_code2"][
            :,
            :,
            code2_st_point[0] : code2_ed_point[0],
            code2_st_point[1] : code2_ed_point[1],
            code2_st_point[2] : code2_ed_point[2],
        ]
        cropped_occl1 = data["padded_occl1"][
            :,
            :,
            code2_st_point[0] : code2_ed_point[0],
            code2_st_point[1] : code2_ed_point[1],
            code2_st_point[2] : code2_ed_point[2],
        ]

        # get tsdf origin relative to [whole] code1 vol
        tsdf_origin = data["tsdf_mask"].nonzero().min(dim=0)[0][2:]
        tsdf_st_point += tsdf_origin
        tsdf_ed_point = tsdf_st_point + crop_tsdf_dim
        cropped_tsdf = data["padded_tsdf"][
            :,
            :,
            tsdf_st_point[0] : tsdf_ed_point[0],
            tsdf_st_point[1] : tsdf_ed_point[1],
            tsdf_st_point[2] : tsdf_ed_point[2],
        ]

        ## debug
        occ_num = (cropped_tsdf.abs() < 1.0).nonzero().shape[0]
        if occ_num < 100000:
            print("[debug] occ num={}".format(occ_num))
            print("[debug] scene={}".format(data["scene"]))
            print("[debug] tsdf_st_point", tsdf_st_point)
            print("[debug] tsdf_ed_point", tsdf_ed_point)
            print("[debug] tsdf_origin", tsdf_origin)
            print("[debug] resample")
            return self.__call__(data)

        # round to fixed size for batch collate
        padded_code1_shape = torch.tensor([i // 8 for i in self.voxel_dim])
        padded_code2_shape = torch.tensor([i * 2 for i in padded_code1_shape])
        padded_tsdf_shape = torch.tensor([i * 4 for i in padded_code2_shape])

        # occl1 will be the mask, so we do not provide mask
        data_dict["padded_tsdf"] = RandomTransform.round_vol(
            cropped_tsdf, 1.000, padded_tsdf_shape
        )
        # todo: provide default code for empty volume
        data_dict["padded_code1"] = RandomTransform.round_vol(
            cropped_code1, 0.0, padded_code1_shape
        )
        data_dict["padded_code2"] = RandomTransform.round_vol(
            cropped_code2, 0.0, padded_code2_shape
        )
        data_dict["padded_occl1"] = RandomTransform.round_vol(
            cropped_occl1, -1000.0, padded_code2_shape
        )
        data_dict["scene"] = data["scene"]
        if "scene_description" in data:
            data_dict["scene_description"] = data["scene_description"]
        data_dict["latent_scale"] = data["latent_scale"]
        return data_dict


class SimpleCropTSDF(object):
    def __init__(self, crop_dim) -> None:
        self.voxel_dim = crop_dim

    def random_crop_volume(self, volume, crop_size, threshold):
        M, N, P = volume.size()
        m, n, p = crop_size

        start_i = torch.randint(0, M - m + 1, (1,))
        start_j = torch.randint(0, N - n + 1, (1,))
        start_k = torch.randint(0, P - p + 1, (1,))

        cropped_volume = volume[
            start_i : start_i + m, start_j : start_j + n, start_k : start_k + p
        ]
        voxel_num = m * n * p
        voxel_origin = torch.tensor([start_i, start_j, start_k])

        if (cropped_volume.abs() < 1.0).sum() / voxel_num > threshold:
            return cropped_volume, voxel_origin
        else:
            return self.random_crop_volume(volume, crop_size, threshold)

    def __call__(self, data) -> Any:
        tsdf_vol = data["tsdf"]["gt"]

        crop, voxel_origin = self.random_crop_volume(tsdf_vol, self.voxel_dim, 0.07)
        new_origin = (
            data["tsdf"]["gt_origin"] + voxel_origin * data["tsdf"]["voxel_size"]
        )
        data["tsdf"]["gt_origin"] = new_origin
        if len(crop.shape) == 3:
            crop = crop.unsqueeze(0)
        data["tsdf"]["gt"] = crop
        return data
