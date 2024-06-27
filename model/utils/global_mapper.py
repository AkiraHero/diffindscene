import torch


class GlobalMapper:
    def __init__(
        self,
        old_origin,
        dim_size,
        voxel_size=0.04,
        default_value=1.0,
        device=torch.device("cpu"),
    ) -> None:
        self.voxel_default_value = default_value
        self.scene_map = torch.ones(dim_size).to(device) * self.voxel_default_value
        self.dim = dim_size
        self.origin = (
            torch.div(old_origin, voxel_size, rounding_mode="floor")
        ) * voxel_size
        self.voxel_size = voxel_size
        self.part_id_info = {}
        self.device = device

    def to(self, device):
        self.scene_map = self.scene_map.to(device)
        self.device = device

    def get_overlap_index(self, get_origin, get_dim):
        get_origin = torch.tensor([*get_origin], dtype=torch.int32)
        get_dim = torch.tensor([*get_dim], dtype=torch.int32)
        self_dim = torch.tensor([*self.dim], dtype=torch.int32)

        if ((get_origin - self_dim) >= 0).any():
            return None
        if ((get_origin + get_dim) <= 0).any():
            return None
        inx_end = torch.min(get_origin + get_dim, self_dim)
        inx_st = torch.max(get_origin, torch.tensor((0, 0, 0)))
        get_inx_st = torch.max(-get_origin, torch.tensor((0, 0, 0)))
        get_inx_ed = torch.min(self_dim - get_origin, get_dim)
        assert (inx_end - inx_st == get_inx_ed - get_inx_st).any()
        return (inx_st, inx_end), (get_inx_st, get_inx_ed)

    def get_scene_map(self):
        return {
            "map": self.scene_map,
            "origin": self.origin,
            "voxel_size": self.voxel_size,
        }

    def update(self, part_id, partial_vol, partial_origin, mode="assign"):
        voxel_origin = (partial_origin - self.origin) / self.voxel_size
        # print(voxel_origin)
        voxel_origin = torch.round(voxel_origin).to(torch.int32)
        part_info = {
            "part_voxel_origin": voxel_origin,
            "part_origin": voxel_origin * self.voxel_size + self.origin,
            "part_dim": partial_vol.shape,
        }
        if part_id not in self.part_id_info:
            self.part_id_info[part_id] = part_info
        else:
            old_part_info = self.part_id_info[part_id]
            for k, v in old_part_info:
                assert part_info[k] == v

        indices = self.get_overlap_index(voxel_origin, partial_vol.shape)
        assert indices is not None
        (a_st, a_ed), (b_st, b_ed) = indices
        if mode == "assign":
            self.scene_map[a_st[0] : a_ed[0], a_st[1] : a_ed[1], a_st[2] : a_ed[2]] = (
                partial_vol[b_st[0] : b_ed[0], b_st[1] : b_ed[1], b_st[2] : b_ed[2]]
            )
        elif mode == "add":
            self.scene_map[
                a_st[0] : a_ed[0], a_st[1] : a_ed[1], a_st[2] : a_ed[2]
            ] += partial_vol[b_st[0] : b_ed[0], b_st[1] : b_ed[1], b_st[2] : b_ed[2]]
        elif mode == "random":
            a = self.scene_map[a_st[0] : a_ed[0], a_st[1] : a_ed[1], a_st[2] : a_ed[2]]
            b = partial_vol[b_st[0] : b_ed[0], b_st[1] : b_ed[1], b_st[2] : b_ed[2]]
            c = torch.rand_like(a)
            d = (a != self.voxel_default_value) * c
            # this option works than the previous average!!!!

            self.scene_map[a_st[0] : a_ed[0], a_st[1] : a_ed[1], a_st[2] : a_ed[2]] = (
                a * d + (1 - d) * b
            )

        else:
            raise NotImplementedError

    def get(self, part_id, default_value=0):
        assert part_id in self.part_id_info
        part_info = self.part_id_info[part_id]
        voxel_origin = part_info["part_voxel_origin"]
        part_dim = part_info["part_dim"]
        indices = self.get_overlap_index(voxel_origin, part_dim)
        assert indices is not None
        (a_st, a_ed), (b_st, b_ed) = indices
        target = torch.ones(part_dim).to(self.device) * default_value
        target[b_st[0] : b_ed[0], b_st[1] : b_ed[1], b_st[2] : b_ed[2]] = (
            self.scene_map[a_st[0] : a_ed[0], a_st[1] : a_ed[1], a_st[2] : a_ed[2]]
        )
        return target

    def get_voxel_part(self, voxel_origin, voxel_dim, default_value):
        voxel_origin = voxel_origin
        part_dim = voxel_dim
        indices = self.get_overlap_index(voxel_origin, part_dim)
        assert indices is not None
        (a_st, a_ed), (b_st, b_ed) = indices
        target = torch.ones(part_dim).to(self.device) * default_value
        target[b_st[0] : b_ed[0], b_st[1] : b_ed[1], b_st[2] : b_ed[2]] = (
            self.scene_map[a_st[0] : a_ed[0], a_st[1] : a_ed[1], a_st[2] : a_ed[2]]
        )
        return target
