# Modified from [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie and Jiaming Sun.

import numpy as np
import torch


def coordinates(voxel_dim, device=torch.device("cuda")):
    """3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


class Compose(object):
    """Apply a list of transforms sequentially"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class ToTensor(object):
    """Convert to torch tensors"""

    def __call__(self, data):
        if "tsdf" in data.keys():
            for k, v in data["tsdf"].items():
                if k == "data_type":
                    continue
                if isinstance(v, list):
                    data["tsdf"][k] = [torch.Tensor(i) for i in v]
                else:
                    data["tsdf"][k] = torch.Tensor(data["tsdf"][k])
        for k in ["intrinsic", "extrinsics"]:
            if k in data.keys():
                data[k] = torch.Tensor(data[k])
        return data


class RandomTransformSpace(object):
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
        using_camera_pose=True,
        random_rot_method="random",
        random_trans_method="min_max",
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

        # each epoch has the same transformation
        self.random_r = None
        self.random_t = None
        self.random_t_method = random_trans_method
        self.random_rot_method = random_rot_method
        assert random_trans_method in ["min_max", "occ_center"]
        assert random_rot_method in ["random", "right-angle"]

    def get_rough_bev_occ(self, gt_tsdf):
        tsdf_thres = 1.0
        occ = abs(gt_tsdf) < tsdf_thres
        return occ.nonzero()

    def __call__(self, data):
        origin = torch.Tensor(data["tsdf"]["gt_origin"])
        gt_tsdf = data["tsdf"]["gt"]

        T = torch.eye(4)
        T[0, 3] = origin[0]
        T[1, 3] = origin[1]
        T[2, 3] = origin[2]

        # construct rotaion matrix about z axis
        if self.random_rotation:
            if self.random_rot_method == "random":
                r = torch.rand(1) * 2 * np.pi
            elif self.random_rot_method == "right-angle":
                r = torch.randint(0, 4, (1,)) * np.pi / 2.0
            else:
                raise NotImplementedError

            self.random_r = r
            # first construct it in 2d so we can rotate bounding corners in the plane
            R = torch.tensor(
                [[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]], dtype=torch.float32
            )
            R_inv = torch.tensor(
                [[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]], dtype=torch.float32
            )

            T[:2, :2] = R

        if self.random_translation:
            rough_occ_coords = (
                origin + self.get_rough_bev_occ(gt_tsdf) * self.voxel_size
            )
            if self.random_t_method == "min_max":
                zmin = rough_occ_coords[:, 2].min()
                zmax = rough_occ_coords[:, 2].max()

                rough_occ_coords = rough_occ_coords[:, :2].transpose(0, 1)  # xy
                rough_occ_coords = R_inv @ rough_occ_coords

                xmin = rough_occ_coords[0].min()
                xmax = rough_occ_coords[0].max()
                ymin = rough_occ_coords[1].min()
                ymax = rough_occ_coords[1].max()

                # randomly sample a crop
                start = torch.Tensor([xmin, ymin, zmin]) - self.padding_start
                # ensure visible area, no truncation on surface
                end = (
                    torch.Tensor([xmax, ymax, zmax]) + self.padding_end
                ) - torch.Tensor(self.voxel_dim) * self.voxel_size

                t = torch.rand([1, 3])
                self.random_t = t
                t = (start * t + (1 - t) * end).reshape(1, 3)

            elif self.random_t_method == "occ_center":
                zmin = rough_occ_coords[:, 2].min() - self.padding_start[2]
                zmax = (
                    rough_occ_coords[:, 2].max()
                    + self.padding_end[2]
                    - self.voxel_dim[2] * self.voxel_size
                )
                z = torch.rand((1,)) * (zmax - zmin) + zmin
                rough_occ_coords = rough_occ_coords[:, :2].transpose(0, 1)  # xy
                rough_occ_coords = R_inv @ rough_occ_coords
                rough_occ_coords = rough_occ_coords.transpose(0, 1)
                occ_num = rough_occ_coords.shape[0]
                inx = torch.randint(0, occ_num, (1,))
                x, y = rough_occ_coords[inx, 0], rough_occ_coords[inx, 1]
                x -= self.voxel_dim[0] * self.voxel_size / 2.0
                y -= self.voxel_dim[1] * self.voxel_size / 2.0
                t = torch.tensor([x, y, z]).reshape(1, 3)
            else:
                raise NotImplementedError

            T[2, 3] = t[0, 2]
            T[:2, 3:] = R @ t.reshape(3, 1)[:2]

        data["partial_tsdf"] = {}
        data["partial_tsdf"]["gt_origin"] = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float, device=T.device
        )
        data["vol_origin_partial"] = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float, device=T.device
        )

        data_type = data["tsdf"]["data_type"]
        if data_type == "tsdf":
            pgt, _ = self.crop_tsdf(
                list(data["tsdf"]["crop_dim"].to(int).numpy()),
                data["tsdf"]["voxel_size"],
                data["tsdf"]["gt_origin"],
                data["tsdf"]["gt"],
                data["vol_origin_partial"],
                T,
            )
        elif data_type == "occ":
            pgt, _ = self.crop_occ(
                list(data["tsdf"]["crop_dim"].to(int).numpy()),
                data["tsdf"]["voxel_size"],
                data["tsdf"]["gt_origin"],
                data["tsdf"]["gt"],
                data["vol_origin_partial"],
                T,
                default_value=0,
                interpolation=True,
            )
        else:
            raise NotImplementedError

        data["partial_tsdf"]["gt"] = pgt
        data["partial_tsdf"]["trans"] = T
        data["partial_tsdf"]["old_origin"] = data["tsdf"]["gt_origin"]
        return data

    @staticmethod
    def crop_tsdf(
        voxel_dim,
        voxel_size,
        old_origin,
        initial_tsdf,
        new_origin_under_T,
        transform,
        align_corners=False,
        default_value=1.0,
        interpolation=True,
    ):
        old_origin = old_origin.view(1, 3)
        x, y, z = voxel_dim
        coords = coordinates(voxel_dim, device=old_origin.device)

        # world coord under T
        world = coords.type(torch.float) * voxel_size + new_origin_under_T.view(3, 1)
        world = torch.cat((world, torch.ones_like(world[:1])), dim=0)

        # get 4 xy corners for debug
        corners = [
            [0, 0, 0, 1],
            [x * voxel_size, 0, 0, 1],
            [0, y * voxel_size, 0, 1],
            [x * voxel_size, y * voxel_size, 0, 1],
        ]
        voxel_cos = []
        for c in corners:
            c = torch.tensor(c, dtype=torch.float32)
            c[:3] = c[:3] + new_origin_under_T
            voxel_co = ((transform @ c)[:3] - old_origin) / voxel_size
            voxel_cos.append(voxel_co)

        # project coordinates under new system to the old system
        world = transform[:3, :] @ world
        coords = (world - old_origin.T) / voxel_size

        # grid sample expects coords in [-1,1]
        coords_world_s = coords.view(3, x, y, z)
        dim_s = list(coords_world_s.shape[1:])
        coords_world_s = coords_world_s.view(3, -1)
        tsdf_s = initial_tsdf

        old_voxel_dim = list(tsdf_s.shape)

        coords_world_s = (
            2 * coords_world_s / (torch.Tensor(old_voxel_dim) - 1).view(3, 1) - 1
        )
        coords_world_s = coords_world_s[[2, 1, 0]].T.view([1] + dim_s + [3])

        # bilinear interpolation near surface,
        # no interpolation along -1,1 boundry
        tsdf_vol = torch.nn.functional.grid_sample(
            tsdf_s.view([1, 1] + old_voxel_dim),
            coords_world_s,
            mode="nearest",
            align_corners=align_corners,
        ).squeeze()
        if interpolation:
            tsdf_vol_bilin = torch.nn.functional.grid_sample(
                tsdf_s.view([1, 1] + old_voxel_dim),
                coords_world_s,
                mode="bilinear",
                align_corners=align_corners,
            ).squeeze()
            mask = tsdf_vol.abs() < 1
            tsdf_vol[mask] = tsdf_vol_bilin[mask]

        # padding_mode='ones' does not exist for grid_sample so replace
        # elements that were on the boarder with 1.
        # voxels beyond full volume (prior to croping) should be marked as empty
        mask = (coords_world_s.abs() >= 1).squeeze(0).any(3)
        tsdf_vol[mask] = default_value
        return tsdf_vol, voxel_cos

    @staticmethod
    def crop_occ(
        voxel_dim,
        voxel_size,
        old_origin,
        initial_tsdf,
        new_origin_under_T,
        transform,
        align_corners=False,
        default_value=1.0,
        interpolation=True,
    ):
        old_origin = old_origin.view(1, 3)
        x, y, z = voxel_dim
        coords = coordinates(voxel_dim, device=old_origin.device)

        # world coord under T
        world = coords.type(torch.float) * voxel_size + new_origin_under_T.view(3, 1)
        world = torch.cat((world, torch.ones_like(world[:1])), dim=0)

        # get 4 xy corners for debug
        corners = [
            [0, 0, 0, 1],
            [x * voxel_size, 0, 0, 1],
            [0, y * voxel_size, 0, 1],
            [x * voxel_size, y * voxel_size, 0, 1],
        ]
        voxel_cos = []
        for c in corners:
            c = torch.tensor(c, dtype=torch.float32)
            c[:3] = c[:3] + new_origin_under_T
            voxel_co = ((transform @ c)[:3] - old_origin) / voxel_size
            voxel_cos.append(voxel_co)

        # project coordinates under new system to the old system
        world = transform[:3, :] @ world
        coords = (world - old_origin.T) / voxel_size

        # grid sample expects coords in [-1,1]
        coords_world_s = coords.view(3, x, y, z)
        dim_s = list(coords_world_s.shape[1:])
        coords_world_s = coords_world_s.view(3, -1)
        tsdf_s = initial_tsdf

        old_voxel_dim = list(tsdf_s.shape)

        coords_world_s = (
            2 * coords_world_s / (torch.Tensor(old_voxel_dim) - 1).view(3, 1) - 1
        )
        coords_world_s = coords_world_s[[2, 1, 0]].T.view([1] + dim_s + [3])

        # bilinear interpolation near surface,
        # no interpolation along -1,1 boundry

        if interpolation:
            tsdf_vol_bilin = torch.nn.functional.grid_sample(
                tsdf_s.view([1, 1] + old_voxel_dim),
                coords_world_s,
                mode="bilinear",
                align_corners=align_corners,
            ).squeeze()
            tsdf_vol_bilin[tsdf_vol_bilin <= 0] = 0
            tsdf_vol_bilin[tsdf_vol_bilin > 0] = 1
            tsdf_vol = tsdf_vol_bilin

        else:
            tsdf_vol = torch.nn.functional.grid_sample(
                tsdf_s.view([1, 1] + old_voxel_dim),
                coords_world_s,
                mode="nearest",
                align_corners=align_corners,
            ).squeeze()

        # padding_mode='ones' does not exist for grid_sample so replace
        # elements that were on the boarder with 1.
        # voxels beyond full volume (prior to croping) should be marked as empty
        mask = (coords_world_s.abs() >= 1).squeeze(0).any(3)
        tsdf_vol[mask] = default_value
        return tsdf_vol, voxel_cos

    def __repr__(self):
        return self.__class__.__name__
