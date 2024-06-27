'''
part of this file is modified from pytorch3d

'''
import numpy as np
import torch

from utils.graphics_utils import cubify
from io import BytesIO
from typing import Optional
import logging
import sys

def _write_ply_header(
    f,
    *,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor],
    verts_normals: Optional[torch.Tensor],
    verts_colors: Optional[torch.Tensor],
    ascii: bool,
    colors_as_uint8: bool,
) -> None:
    """
    Internal implementation for writing header when saving to a .ply file.

    Args:
        f: File object to which the 3D data should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        verts_colors: FloatTensor of shape (V, 3) giving vertex colors.
        ascii: (bool) whether to use the ascii ply format.
        colors_as_uint8: Whether to save colors as numbers in the range
                    [0, 255] instead of float32.
    """
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert faces is None or not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)
    assert verts_normals is None or (
        verts_normals.dim() == 2 and verts_normals.size(1) == 3
    )
    assert verts_colors is None or (
        verts_colors.dim() == 2 and verts_colors.size(1) == 3
    )

    if ascii:
        f.write(b"ply\nformat ascii 1.0\n")
    elif sys.byteorder == "big":
        f.write(b"ply\nformat binary_big_endian 1.0\n")
    else:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
    f.write(f"element vertex {verts.shape[0]}\n".encode("ascii"))
    f.write(b"property float x\n")
    f.write(b"property float y\n")
    f.write(b"property float z\n")
    if verts_normals is not None:
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
    if verts_colors is not None:
        color_ply_type = b"uchar" if colors_as_uint8 else b"float"
        for color in (b"red", b"green", b"blue"):
            f.write(b"property " + color_ply_type + b" " + color + b"\n")
    if len(verts) and faces is not None:
        f.write(f"element face {faces.shape[0]}\n".encode("ascii"))
        f.write(b"property list uchar int vertex_index\n")
    f.write(b"end_header\n")

def _check_faces_indices(
    faces_indices: torch.Tensor, max_index: int, pad_value: Optional[int] = None
) -> torch.Tensor:
    if pad_value is None:
        mask = torch.ones(faces_indices.shape[:-1]).bool()  # Keep all faces
    else:
        mask = faces_indices.ne(pad_value).any(dim=-1)
    if torch.any(faces_indices[mask] >= max_index) or torch.any(
        faces_indices[mask] < 0
    ):
        logging.warn("Faces have invalid indices")
    return faces_indices


def _save_ply(
    f,
    *,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor],
    verts_normals: Optional[torch.Tensor] = None,
    verts_colors: Optional[torch.Tensor] = None,
    ascii: bool = False,
    decimal_places: Optional[int] = None,
    colors_as_uint8: bool = False,
) -> None:
    """
    Internal implementation for saving 3D data to a .ply file.

    Args:
        f: File object to which the 3D data should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        verts_colors: FloatTensor of shape (V, 3) giving vertex colors.
        ascii: (bool) whether to use the ascii ply format.
        decimal_places: Number of decimal places for saving if ascii=True.
        colors_as_uint8: Whether to save colors as numbers in the range
                    [0, 255] instead of float32.
    """
    _write_ply_header(
        f,
        verts=verts,
        faces=faces,
        verts_normals=verts_normals,
        verts_colors=verts_colors,
        ascii=ascii,
        colors_as_uint8=colors_as_uint8,
    )

    if not (len(verts)):
        logging.warn("Empty 'verts' provided")
        return

    color_np_type = np.ubyte if colors_as_uint8 else np.float32
    verts_dtype = [("verts", np.float32, 3)]
    if verts_normals is not None:
        verts_dtype.append(("normals", np.float32, 3))
    if verts_colors is not None:
        verts_dtype.append(("colors", color_np_type, 3))

    vert_data = np.zeros(verts.shape[0], dtype=verts_dtype)
    vert_data["verts"] = verts.detach().cpu().numpy()
    if verts_normals is not None:
        vert_data["normals"] = verts_normals.detach().cpu().numpy()
    if verts_colors is not None:
        color_data = verts_colors.detach().cpu().numpy()
        if colors_as_uint8:
            vert_data["colors"] = np.rint(color_data * 255)
        else:
            vert_data["colors"] = color_data

    if ascii:
        if decimal_places is None:
            float_str = b"%f"
        else:
            float_str = b"%" + b".%df" % decimal_places
        float_group_str = (float_str + b" ") * 3
        formats = [float_group_str]
        if verts_normals is not None:
            formats.append(float_group_str)
        if verts_colors is not None:
            formats.append(b"%d %d %d " if colors_as_uint8 else float_group_str)
        formats[-1] = formats[-1][:-1] + b"\n"
        for line_data in vert_data:
            for data, format in zip(line_data, formats):
                f.write(format % tuple(data))
    else:
        if isinstance(f, BytesIO):
            # tofile only works with real files, but is faster than this.
            f.write(vert_data.tobytes())
        else:
            vert_data.tofile(f)

    if faces is not None:
        faces_array = faces.detach().cpu().numpy()

        _check_faces_indices(faces, max_index=verts.shape[0])

        if len(faces_array):
            if ascii:
                np.savetxt(f, faces_array, "3 %d %d %d")
            else:
                faces_recs = np.zeros(
                    len(faces_array),
                    dtype=[("count", np.uint8), ("vertex_indices", np.uint32, 3)],
                )
                faces_recs["count"] = 3
                faces_recs["vertex_indices"] = faces_array
                faces_uints = faces_recs.view(np.uint8)

                if isinstance(f, BytesIO):
                    f.write(faces_uints.tobytes())
                else:
                    faces_uints.tofile(f)



def occ2mesh(occ_vol, thres=0.9, voxel_size=0.04, origin=torch.tensor((0,0,0))):
    if len(occ_vol.shape) < 4:
        occ_vol = occ_vol.unsqueeze(0)
    meshvert, face = cubify(occ_vol, thres, align='center')
    verts=meshvert[0]
    faces=face[0]
    if not verts.shape[0]:
        return None, None
    
    origin = origin.to(verts.device)
    verts = verts * voxel_size + origin

    # verts[:, [0,1]] = verts[:, [1,0]]
    # faces[:, [0,1,2]] = faces[:, [0,2,1]]

    verts[:, [0,2]] = verts[:, [2,0]]
    faces[:, [0,1,2]] = faces[:, [2,1,0]]
    
    return verts, faces


def mesh2ply(verts, faces, filename):
    with open(filename, 'wb') as f:
        _save_ply(f, verts=verts, faces=faces)


