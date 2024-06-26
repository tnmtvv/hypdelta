from typing import Tuple
import numpy as np
import math
from numba import njit, prange


@njit(fastmath=True)
def indx_to_2d(k: int) -> Tuple[int, int]:
    """
    Converts a 1D index to a 2D index following a specific pattern.

    Parameters:
    -----------
    k : int
        The 1D index to be converted.

    Returns:
    --------
    Tuple[int, int]
        The corresponding 2D index (n, m).
    """
    # Find n using the quadratic formula component
    n = math.floor((1 + math.sqrt(1 + 8 * k)) / 2)

    # Calculate the corresponding triangular number T_{n-1}
    T_n_minus_1 = (n * (n - 1)) // 2

    # Calculate m
    m = k - T_n_minus_1

    return (n, m)


def calc_max_lines(gpu_mem_bound: float, pairs_len: int) -> int:
    """
    Calculates the maximum number of lines that can be processed based on GPU memory constraints.

    Parameters:
    -----------
    gpu_mem_bound : float
        The GPU memory limit in gigabytes.
    pairs_len : int
        The number of points (or pairs) to consider.

    Returns:
    --------
    int
        The maximum number of lines that can be processed given the GPU memory constraints.

    Prints:
    -------
    cartesian_size : int
        The total number of pairs (size of the Cartesian product).
    parts : float
        The number of parts the dataset is divided into based on the memory constraint.
    """
    cartesian_size = int(pairs_len * (pairs_len - 1) / 2)
    parts = (cartesian_size * 6 * 8) / (gpu_mem_bound * math.pow(10, 9))
    print(cartesian_size)
    print(parts)
    max_lines = int(np.ceil(cartesian_size / parts))
    return max_lines


@njit(parallel=True)
def batch_flatten(indices: np.ndarray, dist_matrix_flat: np.ndarray) -> np.ndarray:
    """
    Extracts and returns a batch of values from a flattened distance matrix
    based on the provided indices, using parallel computation for efficiency.

    Parameters:
    -----------
    indices : np.ndarray
        An array of indices used to extract values from the flattened distance matrix.
    dist_matrix_flat : np.ndarray
        A 1-dimensional flattened distance matrix from which values are to be extracted.

    Returns:
    --------
    np.ndarray
        An array containing the values extracted from the flattened distance matrix
        at the specified indices.
    """
    num = indices.shape[0]
    batch = np.zeros(indices.shape[0], dtype="double")

    for i in prange(0, num):
        batch[i] = dist_matrix_flat[indices[i]]
    return batch


@njit(parallel=True)
def prepare_batch_indices_flat(
    far_away_pairs: list, start_ind: int, end_ind: int, n: int
) -> np.ndarray:
    """
    Prepares a flattened array of batch indices for Cartesian products.

    Parameters:
    -----------
    far_away_pairs : list of tuples
        List of far away pairs.
    start_ind : int
        The starting index for the batch.
    end_ind : int
        The ending index for the batch.
    n : int
        The dimension size of the original distance matrix.

    Returns:
    --------
    np.ndarray
        A flattened array of indices for the batch.
    """
    batch_indices = np.empty(int(end_ind - start_ind) * 6, dtype=np.int32)

    for indx in prange(start_ind, end_ind):
        i, j = indx_to_2d(indx)

        pair_1 = far_away_pairs[i]
        pair_2 = far_away_pairs[j]

        batch_indices[6 * (indx - start_ind) + 0] = pair_1[0] * n + pair_1[1]
        batch_indices[6 * (indx - start_ind) + 1] = pair_2[0] * n + pair_2[1]
        batch_indices[6 * (indx - start_ind) + 2] = pair_1[0] * n + pair_2[0]
        batch_indices[6 * (indx - start_ind) + 3] = pair_1[1] * n + pair_2[1]
        batch_indices[6 * (indx - start_ind) + 4] = pair_2[0] * n + pair_1[1]
        batch_indices[6 * (indx - start_ind) + 5] = pair_2[1] * n + pair_1[0]
    return batch_indices


def get_far_away_pairs(A: np.ndarray, N: int) -> list:
    """
    Identifies pairs of points that are far away in the distance matrix.

    Parameters:
    -----------
    A : np.ndarray
        The distance matrix.
    N : int
        The number of far away pairs to retrieve.

    Returns:
    --------
    list of tuples
        A list of tuples representing the indices of far away pairs.
    """
    a = -A.ravel()
    N = min(a.shape[0] - 1, N)
    a_indx = np.argpartition(a, N)
    indx_sorted = zip(
        *np.unravel_index(sorted(a_indx[: N + 1], key=lambda i: a[i]), A.shape)
    )
    return [(i, j) for (i, j) in indx_sorted if i < j]


@njit()
def s_delta(
    dist: np.ndarray, ind_i: int, ind_j: int, k: int, delta_hyp_k: float
) -> float:
    """
    Calculate the delta hyperbolicity for a given set of distances.

    Parameters:
    -----------
    dist : np.ndarray
        A 2D array representing the distance matrix. dist[i][j] is the distance between points i and j.
    ind_i : int
        The index of the first point in the pair (i, j).
    ind_j : int
        The index of the second point in the pair (i, j).
    k : int
        The index of the third point k.
    delta_hyp_k : float
        The current maximum delta hyperbolicity value.

    Returns:
    --------
    float
        The updated delta hyperbolicity value.

    Notes:
    ------
    This function uses the definition of delta hyperbolicity which is the maximum of the difference between
    the two largest values among the sums of the distances for each triple (i, j, k).
    """
    dist_0k = dist[0][k - 1]
    dist_0i = dist[0][ind_i]
    dist_ik = dist[ind_i][k - 1]

    dist_0j = dist[0][ind_j]
    dist_jk = dist[ind_j][k - 1]
    dist_ij = dist[ind_i][ind_j]

    # Compute the distances
    dist_array = [dist_0j + dist_ik, dist_0i + dist_jk, dist_0k + dist_ij]
    s2, s1 = sorted(dist_array)[-2:]

    # Update delta_hyp_k with the maximum difference between the two largest sums
    delta_hyp_k = max(delta_hyp_k, s1 - s2)
    return max(delta_hyp_k, s1 - s2)
