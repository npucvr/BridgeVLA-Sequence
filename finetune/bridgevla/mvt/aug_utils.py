# copy from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/aug_utils.py
import torch
import numpy as np
from scipy.spatial.transform import Rotation


def rand_dist(size, min=-1.0, max=1.0):
    return (max - min) * torch.rand(size) + min


def rand_discrete(size, min=0, max=1):
    if min == max:
        return torch.zeros(size)
    return torch.randint(min, max + 1, size)


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def sensitive_gimble_fix(euler):
    """
    :param euler: euler angles in degree as np.ndarray in shape either [3] or
    [b, 3]
    """
    # selecting sensitive angle
    select1 = (89 < euler[..., 1]) & (euler[..., 1] < 91)
    euler[select1, 1] = 90
    # selecting sensitive angle
    select2 = (-91 < euler[..., 1]) & (euler[..., 1] < -89)
    euler[select2, 1] = -90

    # recalulating the euler angles, see assert
    r = Rotation.from_euler("xyz", euler, degrees=True)
    euler = r.as_euler("xyz", degrees=True)

    select = select1 | select2
    assert (euler[select][..., 2] == 0).all(), euler

    return euler


def quaternion_to_discrete_euler(quaternion, resolution, gimble_fix=True):
    """
    :param gimble_fix: the euler values for x and y can be very sensitive
        around y=90 degrees. this leads to a multimodal distribution of x and y
        which could be hard for a network to learn. When gimble_fix is true, around
        y=90, we change the mode towards x=0, potentially making it easy for the
        network to learn.
    """
    r = Rotation.from_quat(quaternion)

    euler = r.as_euler("xyz", degrees=True)
    if gimble_fix:
        euler = sensitive_gimble_fix(euler)

    euler += 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def quaternion_to_euler(quaternion, gimble_fix=True):
    """
    :param gimble_fix: the euler values for x and y can be very sensitive
        around y=90 degrees. this leads to a multimodal distribution of x and y
        which could be hard for a network to learn. When gimble_fix is true, around
        y=90, we change the mode towards x=0, potentially making it easy for the
        network to learn.
    """
    r = Rotation.from_quat(quaternion)

    euler = r.as_euler("xyz", degrees=True)
    if gimble_fix:
        euler = sensitive_gimble_fix(euler)

    euler += 180
    return euler


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


# 6D continuous rotation representation (Zhou et al., CVPR 2019,
# "On the Continuity of Rotation Representations in Neural Networks").
# Ported from BridgeVLA++ (main) for rot_ver == 2. These ops are pure-torch
# and differentiable so ``rotation_6d_to_matrix`` can sit inside the training
# loss. Convention: the 6D vector packs the first two COLUMNS of the rotation
# matrix; Gram-Schmidt recovers an orthonormal R in SO(3). There is NO
# discretization, NO gimble_fix, and NO quaternion double-cover handling
# (q and -q map to the same R, so the sign is irrelevant here).


def rotation_6d_to_matrix(d6):
    """(..., 6) -> (..., 3, 3) rotation matrix via Gram-Schmidt.

    a1 = d6[..., 0:3], a2 = d6[..., 3:6] are the (unconstrained) first two
    columns. b1 = normalize(a1); b2 = normalize(a2 - <b1,a2> b1);
    b3 = b1 x b2. The three orthonormal vectors are stacked as COLUMNS.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix):
    """(..., 3, 3) -> (..., 6): the first two COLUMNS flattened.

    Inverse companion of :func:`rotation_6d_to_matrix` (up to the
    Gram-Schmidt projection, which is the identity for a valid R).
    """
    return torch.cat((matrix[..., 0], matrix[..., 1]), dim=-1)


def quaternion_xyzw_to_matrix_np(quat_xyzw):
    """(..., 4) numpy quaternion (xyzw) -> (..., 3, 3) numpy rotation matrix."""
    return Rotation.from_quat(np.asarray(quat_xyzw)).as_matrix()


def matrix_to_quaternion_xyzw_np(matrix):
    """(..., 3, 3) numpy rotation matrix -> (..., 4) numpy quaternion (xyzw)."""
    return Rotation.from_matrix(np.asarray(matrix)).as_quat()


def point_to_voxel_index(
    point: np.ndarray, voxel_size: np.ndarray, coord_bounds: np.ndarray
):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one
    )
    return voxel_indicy
