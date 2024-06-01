import numpy as np
from numba import njit, prange

from hypdelta.calculus_utils import *


@njit(parallel=True)
def calculate_delta_condensed(dist_condensed: np.ndarray, n_samples: int) -> float:
    """
    Compute the delta hyperbolicity value from the condensed distance matrix representation.
    This is a more efficient analog of the `delta_hyp` function.

    Parameters
    ----------
    dist_condensed : numpy.ndarray
        A 1D array representing the condensed distance matrix of the dataset.
    n_samples : int
        The number of nodes in the dataset.

    Returns
    -------
    float
        The delta hyperbolicity of the dataset.

    Notes
    -----
    Calculation heavily relies on the `scipy`'s `pdist` output format. According to the docs (as of v.1.10.1):
    "The metric dist(u=X[i], v=X[j]) is computed and stored in entry m * i + j - ((i + 2) * (i + 1)) // 2."
    Additionally, it implicitly assumes that j > i. Note that dist(u=X[0], v=X[k]) is defined by (k - 1)'s entry.
    """
    delta_hyp = np.zeros(n_samples, dtype=dist_condensed.dtype)
    for k in prange(n_samples):
        # as in `delta_hyp`, fixed point is selected at 0
        delta_hyp_k = 0.0
        dist_0k = 0.0 if k == 0 else dist_condensed[k - 1]
        for i in range(n_samples):
            if i == 0:
                dist_0i = 0.0
                dist_ik = dist_0k
            else:
                if k == i:
                    dist_0i = dist_0k
                    dist_ik = 0.0
                else:
                    dist_0i = dist_condensed[i - 1]
                    i1, i2 = (i, k) if k > i else (k, i)
                    ik_idx = n_samples * i1 + i2 - ((i1 + 2) * (i1 + 1)) // 2
                    dist_ik = dist_condensed[int(ik_idx)]
            diff_ik = dist_0i - dist_ik
            for j in range(i, n_samples):
                if j == 0:
                    dist_0j = 0.0
                    dist_jk = dist_0k
                else:
                    if k == j:
                        dist_0j = dist_0k
                        dist_jk = 0.0
                    else:
                        dist_0j = dist_condensed[j - 1]
                        j1, j2 = (j, k) if k > j else (k, j)
                        jk_idx = n_samples * j1 + j2 - ((j1 + 2) * (j1 + 1)) // 2
                        dist_jk = dist_condensed[int(jk_idx)]
                diff_jk = dist_0j - dist_jk
                if i == j:
                    dist_ij = 0.0
                else:
                    ij_idx = (
                        n_samples * i + j - ((i + 2) * (i + 1)) // 2
                    )  # j >= i by construction
                    dist_ij = dist_condensed[int(ij_idx)]
                gromov_ij = dist_0i + dist_0j - dist_ij
                delta_hyp_k = max(
                    delta_hyp_k, dist_0k + min(diff_ik, diff_jk) - gromov_ij
                )
        delta_hyp[k] = delta_hyp_k
    return 0.5 * np.max(delta_hyp)


@njit(parallel=True)
def calculate_delta_heuristic(dist_matrix: np.ndarray) -> float:
    """
    Compute the delta hyperbolicity value from the condensed distance matrix representation.
    This is a modified version of the `delta_hyp_condenced` function.

    Parameters
    ----------
    dist_matrix : numpy.ndarray
        A 1D array representing the condensed distance matrix of the dataset.
    n_samples : int
        The number of nodes in the dataset.
    const : int
        Number of most distant points that are considered by the algo.
    seed : int
        Seed for experiments reproductibility.
    mode : str
        Mode of function execution.

    Returns
    -------
    float
        The delta hyperbolicity of the dataset.

    Notes
    -----
    The idea is that we can select points partly randomly to achieve a better covering of an item space.
    """
    items = dist_matrix.shape[0]
    delta_hyp = np.zeros(items, dtype=dist_matrix.dtype)
    const = min(50, dist_matrix.shape[0] - 1)

    for k in prange(items):
        delta_hyp_k = 0.0
        inds_i = np.argpartition(dist_matrix[k - 1], -const)
        considered_i = inds_i[-const:]

        for ind_i in considered_i:
            inds_j = np.argpartition(dist_matrix[ind_i - 1], -const)
            considered_j = inds_j[-const:]

            for ind_j in considered_j:
                delta_hyp_k = s_delta(dist_matrix, ind_i, ind_j, k, delta_hyp_k)
        delta_hyp[k] = delta_hyp_k
    return 0.5 * np.max(delta_hyp)


def delta_condensed(dist_matrix, tries, heuristic):
    diam = np.max(dist_matrix)
    if heuristic == True:
        delta = calculate_delta_heuristic(dist_matrix)
    else:
        delta = calculate_delta_condensed(dist_matrix, tries)
    return 2 * delta / diam
