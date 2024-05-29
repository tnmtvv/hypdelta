import numpy as np
from numba import njit, prange, typed, cuda
from typing import List, Tuple, Dict

from hyppy.calculus_utils import get_far_away_pairs, prepare_batch_indices_flat
from hyppy.cudaprep import cuda_prep_CCL, cuda_prep_cartesian


def delta_CCL_cpu(dist_matrix: np.ndarray, l: float) -> Tuple[float, float]:
    """
    Computes the delta hyperbolicity using the CCL method on the CPU.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        The distance matrix.
    l : float
        A factor determining the number of far-away pairs to consider.

    Returns:
    --------
    Tuple[float, float]
        The computed delta hyperbolicity and the diameter of the distance matrix.
    """
    n = dist_matrix.shape[0]
    diam = np.max(dist_matrix)
    far_away_pairs = get_far_away_pairs(dist_matrix, int((n * (n + 1) / 2) * l))
    delta = delta_hyp_CCL(typed.List(far_away_pairs), dist_matrix)
    return 2 * delta / diam, diam


def delta_CCL_gpu(
    dist_matrix: np.ndarray,
    l: float,
    threadsperblock: Tuple[int, int] = (32, 32),
) -> Tuple[float, float]:
    """
    Compute the delta-hyperbolicity of a distance matrix using GPU acceleration.

    Parameters:
    ----------
    dist_matrix : np.ndarray
        A 2D numpy array representing the pairwise distance matrix.

    l : float
        A factor determining the number of far-away pairs to consider.

    threadsperblock : Tuple[int, int], optional
        A tuple specifying the number of threads per block to use in the GPU computation.
        Default is (32, 32).

    Returns:
    -------
    delta : float
        The computed delta-hyperbolicity value.

    diam : float
        The diameter of the graph represented by the distance matrix, i.e., the maximum distance between any two nodes.

    Notes:
    -----
    This function uses CUDA for GPU acceleration to compute the delta-hyperbolicity, which significantly speeds up
    the computation compared to a CPU-based approach. The input distance matrix should be square and symmetric.

    Example:
    --------
    >>> dist_matrix = np.array([[0, 1, 2],
    >>>                         [1, 0, 1],
    >>>                         [2, 1, 0]])
    >>> l = {'threshold': 0.5}
    >>> delta, diam = CCL_gpu(dist_matrix, l)
    >>> print(delta, diam)
    0.6666666666666666, 2.0
    """
    diam = np.max(dist_matrix)
    n = dist_matrix.shape[0]

    far_away_pairs = get_far_away_pairs(int((n * (n + 1) / 2) * l))
    (
        n,
        x_coord_pairs,
        y_coord_pairs,
        adj_m,
        blockspergrid,
        threadsperblock,
        delta_res,
    ) = cuda_prep_CCL(far_away_pairs, dist_matrix, threadsperblock)
    delta_hyp_CCL_GPU[blockspergrid, threadsperblock](
        n, x_coord_pairs, y_coord_pairs, adj_m, delta_res
    )
    delta, _ = 2 * delta_res[0] / diam
    return delta, diam


@njit(parallel=True, fastmath=True)
def delta_hyp_CCL(far_apart_pairs: List[Tuple[int, int]], adj_m: np.ndarray) -> float:
    """
    Computes the delta hyperbolicity for a set of far apart pairs on the CPU.

    Parameters:
    -----------
    far_apart_pairs : List[Tuple[int, int]]
        List of far apart pairs.
    adj_m : np.ndarray
        The distance matrix.

    Returns:
    --------
    float
        The computed delta hyperbolicity.
    """
    delta_hyp = 0.0
    for iter_1 in prange(1, len(far_apart_pairs)):
        pair_1 = far_apart_pairs[iter_1]
        for iter_2 in prange(iter_1):
            pair_2 = far_apart_pairs[iter_2]
            i, j = pair_1
            v, w = pair_2

            d_ij = adj_m[i][j]
            d_iw = adj_m[i][w]
            d_iv = adj_m[i][v]
            d_jw = adj_m[j][w]
            d_jv = adj_m[j][v]
            d_vw = adj_m[v][w]

            cur_del = (d_ij + d_vw - max(d_iv + d_jw, d_iw + d_jv)) / 2
            delta_hyp = max(delta_hyp, cur_del)

    return delta_hyp


@cuda.jit
def delta_hyp_CCL_GPU(
    n: int,
    fisrt_points: cuda.devicearray.DeviceNDArray,
    second_points: cuda.devicearray.DeviceNDArray,
    adj_m: cuda.devicearray.DeviceNDArray,
    delta_res: cuda.devicearray.DeviceNDArray,
) -> None:
    """
    Computes the delta hyperbolicity for a set of far apart pairs on the GPU.

    Parameters:
    -----------
    n : int
        The number of far apart pairs.
    fisrt_points : cuda.devicearray.DeviceNDArray
        The first points of the pairs.
    second_points : cuda.devicearray.DeviceNDArray
        The second points of the pairs.
    adj_m : cuda.devicearray.DeviceNDArray
        The distance matrix.
    delta_res : cuda.devicearray.DeviceNDArray
        The result array to store the maximum delta value.
    """
    n_samples = n
    row, col = cuda.grid(2)

    if row < n_samples:
        i = fisrt_points[row]
        j = second_points[row]
        if col < row:
            v = fisrt_points[col]
            w = second_points[col]

            d_ij = adj_m[i][j]
            d_iw = adj_m[i][w]
            d_iv = adj_m[i][v]
            d_jw = adj_m[j][w]
            d_jv = adj_m[j][v]
            d_vw = adj_m[v][w]

            cuda.atomic.max(
                delta_res, 0, (d_ij + d_vw - max(d_iv + d_jw, d_iw + d_jv)) / 2
            )
