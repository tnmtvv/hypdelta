from typing import Tuple
from sklearn.metrics import pairwise_distances
import numpy as np
from numba import cuda
from hypdelta.cudaprep import cuda_prep_true_delta


def naive_cpu(dist_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Computes the naive delta and diameter of a distance matrix using CPU.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        The distance matrix.

    Returns:
    --------
    Tuple[float, float]
        A tuple containing the delta and the diameter.
    """
    delta = 0
    diam = np.max(dist_matrix)
    for p in range(dist_matrix.shape[0]):
        row = dist_matrix[p, :][np.newaxis, :]
        col = dist_matrix[:, p][:, np.newaxis]
        XY_p = 0.5 * (row + col - dist_matrix)
        maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
        delta = max(delta, np.max(maxmin - XY_p))
    return 2 * delta / diam, diam


def naive_gpu(dist_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Computes the naive delta and diameter of a distance matrix using GPU.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        The distance matrix.

    Returns:
    --------
    Tuple[float, float]
        A tuple containing the delta and the diameter.
    """
    diam = np.max(dist_matrix)
    adj_m, k, delta_res, threadsperblock, blockspergrid = cuda_prep_true_delta(
        dist_matrix
    )
    true_delta_gpu[blockspergrid, threadsperblock](adj_m, delta_res, k)
    delta = delta_res[0]
    return 2 * delta / diam, diam


@cuda.jit
def true_delta_gpu(
    dismat: np.ndarray, delta_res: np.ndarray, num_arr: np.ndarray, diff_num: np.ndarray
) -> None:
    """
    CUDA kernel function to compute the true delta value on the GPU.

    Parameters:
    -----------
    dismat : np.ndarray
        The distance matrix.
    delta_res : np.ndarray
        The array to store the resulting delta value.
    num_arr : np.ndarray
        An array containing configuration numbers.
    diff_num : np.ndarray
        An array containing the difference number.
    """
    num_1 = num_arr[0]
    num_2 = num_arr[1]
    d_m_len = num_arr[2]
    diff_num = diff_num[0]

    h = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) & num_1
    k = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) >> (num_2 - diff_num)
    c = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    p = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

    if p < d_m_len and c < d_m_len and h < d_m_len and k < d_m_len:
        cuda.atomic.max(
            delta_res,
            (0),
            0.5
            * (
                dismat[p * d_m_len + h]
                + dismat[k * d_m_len + c]
                - max(
                    dismat[p * d_m_len + k] + dismat[h * d_m_len + c],
                    dismat[p * d_m_len + c] + dismat[k * d_m_len + h],
                )
            ),
        )
