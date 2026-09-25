"""Small, dependency-free rotation conversions used by the continuous head.

The RLBench action interface stores quaternions in scalar-last ``xyzw``
format.  The 6D representation follows the continuous rotation convention:
the first two columns of a rotation matrix are predicted and the third column
is reconstructed with Gram--Schmidt orthogonalization.
"""

import torch
import torch.nn.functional as F


def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 4)`` scalar-last quaternions to rotation matrices."""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return torch.stack(
        (
            1 - 2 * (yy + zz),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (xx + zz),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (xx + yy),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quaternion_xyzw_to_ortho6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-last quaternions to the first two matrix columns."""
    matrix = quaternion_xyzw_to_matrix(quaternion)
    return matrix[..., :, :2].transpose(-2, -1).reshape(*quaternion.shape[:-1], 6)


def ortho6d_to_matrix(ortho6d: torch.Tensor) -> torch.Tensor:
    """Reconstruct a proper rotation matrix from a 6D representation."""
    first = F.normalize(ortho6d[..., 0:3], dim=-1, eps=1e-6)
    second_raw = ortho6d[..., 3:6]
    second = second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized scalar-last quaternions."""
    original_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    q = torch.zeros((m.shape[0], 4), dtype=m.dtype, device=m.device)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    trace_mask = trace > 0
    if trace_mask.any():
        s = (trace[trace_mask] + 1).clamp_min(1e-8).sqrt() * 2
        mt = m[trace_mask]
        q[trace_mask, 0] = (mt[:, 2, 1] - mt[:, 1, 2]) / s
        q[trace_mask, 1] = (mt[:, 0, 2] - mt[:, 2, 0]) / s
        q[trace_mask, 2] = (mt[:, 1, 0] - mt[:, 0, 1]) / s
        q[trace_mask, 3] = 0.25 * s

    remaining = ~trace_mask
    x_mask = remaining & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    if x_mask.any():
        s = (1 + m[x_mask, 0, 0] - m[x_mask, 1, 1] - m[x_mask, 2, 2]).clamp_min(1e-8).sqrt() * 2
        mx = m[x_mask]
        q[x_mask, 0] = 0.25 * s
        q[x_mask, 1] = (mx[:, 0, 1] + mx[:, 1, 0]) / s
        q[x_mask, 2] = (mx[:, 0, 2] + mx[:, 2, 0]) / s
        q[x_mask, 3] = (mx[:, 2, 1] - mx[:, 1, 2]) / s

    y_mask = remaining & ~x_mask & (m[:, 1, 1] >= m[:, 2, 2])
    if y_mask.any():
        s = (1 - m[y_mask, 0, 0] + m[y_mask, 1, 1] - m[y_mask, 2, 2]).clamp_min(1e-8).sqrt() * 2
        my = m[y_mask]
        q[y_mask, 0] = (my[:, 0, 1] + my[:, 1, 0]) / s
        q[y_mask, 1] = 0.25 * s
        q[y_mask, 2] = (my[:, 1, 2] + my[:, 2, 1]) / s
        q[y_mask, 3] = (my[:, 0, 2] - my[:, 2, 0]) / s

    z_mask = remaining & ~x_mask & ~y_mask
    if z_mask.any():
        s = (1 - m[z_mask, 0, 0] - m[z_mask, 1, 1] + m[z_mask, 2, 2]).clamp_min(1e-8).sqrt() * 2
        mz = m[z_mask]
        q[z_mask, 0] = (mz[:, 0, 2] + mz[:, 2, 0]) / s
        q[z_mask, 1] = (mz[:, 1, 2] + mz[:, 2, 1]) / s
        q[z_mask, 2] = 0.25 * s
        q[z_mask, 3] = (mz[:, 1, 0] - mz[:, 0, 1]) / s

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.where(q[:, 3:4] < 0, -q, q)
    return q.reshape(*original_shape, 4)


def ortho6d_to_quaternion_xyzw(ortho6d: torch.Tensor) -> torch.Tensor:
    """Convert a continuous 6D rotation prediction to scalar-last quaternion."""
    return matrix_to_quaternion_xyzw(ortho6d_to_matrix(ortho6d))
