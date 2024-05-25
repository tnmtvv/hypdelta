import numpy as np
import math
from numba import njit, prange


@njit()
def indx_to_2d(indx):
    """
    Converts a linear index to a 2D index of a cartesian product.

    Parameters:
    -----------
    indx : int
        The linear index to be converted.

    Returns:
    --------
    tuple
        A tuple (n, k) representing the 2D index.
    """
    n = round(math.sqrt(2 * indx))
    S_n = (1 + n) / 2 * n
    return n, int(n - (S_n - indx) - 1)


@njit(parallel=True)
def prepare_batch_indices_flat(far_away_pairs, start_ind, end_ind, n):
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


def get_far_away_pairs(A, N):
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
