"""
Utils used in visualization
"""
import brax
import jax.numpy as jnp
from brax import QP


def deserialize_qp(nparray) -> brax.QP:
    """
    Get QP from a trajectory numpy array
    """
    num_bodies = nparray.shape[-1] // 13  # pos (,3) rot (,4) vel (,3) ang (,3)
    batch_dims = nparray.shape[:-1]
    slices = [num_bodies * x for x in [0, 3, 7, 10, 13]]
    pos = jnp.reshape(nparray[..., slices[0] : slices[1]], batch_dims + (num_bodies, 3))
    rot = jnp.reshape(nparray[..., slices[1] : slices[2]], batch_dims + (num_bodies, 4))
    vel = jnp.reshape(nparray[..., slices[2] : slices[3]], batch_dims + (num_bodies, 3))
    ang = jnp.reshape(nparray[..., slices[3] : slices[4]], batch_dims + (num_bodies, 3))
    return QP(pos=pos, rot=rot, vel=vel, ang=ang)


def serialize_qp(qp) -> jnp.array:
    """
    Serialize QP to a trajectory numpy array
    """
    pos = qp.pos
    rot = qp.rot
    vel = qp.vel
    ang = qp.ang
    batch_dim = pos.shape[:-2]
    nparray = []
    nparray.append(pos.reshape(batch_dim + (-1,)))
    nparray.append(rot.reshape(batch_dim + (-1,)))
    nparray.append(vel.reshape(batch_dim + (-1,)))
    nparray.append(ang.reshape(batch_dim + (-1,)))
    return jnp.concatenate(nparray, axis=-1)


def quaternion_to_matrix(quaternions):
    """
    Converts Quaternions to a 3x3 matrix
    """
    r, i, j, k = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = jnp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix):
    """
    Converts 3x3 rotation matrix to 6D rotation matrix
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def quaternion_to_rotation_6d(quaternion):
    """
    Converts qauternions to 6D rotation matrix
    """
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))
