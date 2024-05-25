import numpy as np
import math
from numba import njit, prange


def get_far_away_pairs(A, N):
    a = -A.ravel()
    N = min(a.shape[0] - 1, N)
    a_indx = np.argpartition(a, N)
    indx_sorted = zip(
        *np.unravel_index(sorted(a_indx[: N + 1], key=lambda i: a[i]), A.shape)
    )
    return [(i, j) for (i, j) in indx_sorted if i < j]


@njit()
def indx_to_2d(indx):
    n = round(math.sqrt(2 * indx))
    S_n = (1 + n) / 2 * n
    return n, int(n - (S_n - indx) - 1)


@njit(parallel=True)
def batch_flatten(indices, dist_matrix_flat):
    num = indices.shape[0]
    batch = np.zeros(indices.shape[0], dtype="double")

    for i in range(0, num):
        batch[i] = dist_matrix_flat[indices[i]]
    return batch


@njit(parallel=True)
def prepare_batch_indices_flat(far_away_pairs, start_ind, end_ind, n):
    batch_indices = np.empty(int(end_ind - start_ind) * 6, dtype=np.int32)

    for indx in prange(start_ind, end_ind):
        i, j = indx_to_2d(indx)

        pair_1 = far_away_pairs[i]
        pair_2 = far_away_pairs[j]

        batch_indices[6 * (indx - start_ind) + 0] = pair_1[0] * n + pair_1[1]
        batch_indices[6 * (indx - start_ind) + 1] = pair_2[0] * n + pair_2[1]
        batch_indices[6 * (indx - start_ind) + 2] = pair_1[0] * n + pair_2[0]
        batch_indices[6 * (indx - start_ind) + 3] = pair_1[1] * n + pair_2[1]
        batch_indices[6 * (indx - start_ind) + 4] = pair_1[0] * n + pair_2[0]
        batch_indices[6 * (indx - start_ind) + 5] = pair_1[1] * n + pair_2[1]
    return batch_indices


@njit
def s_delta(dist, ind_i, ind_j, k, delta_hyp_k):
    dist_0k = dist[0][k - 1]
    dist_0i = dist[0][ind_i]
    dist_ik = dist[ind_i][k - 1]

    dist_0j = dist[0][ind_j]
    dist_jk = dist[ind_j][k - 1]
    dist_ij = dist[ind_i][ind_j]

    # algo with S
    dist_array = [dist_0j + dist_ik, dist_0i + dist_jk, dist_0k + dist_ij]
    s2, s1 = sorted(dist_array)[-2:]
    delta_hyp_k = max(delta_hyp_k, s1 - s2)
    return max(delta_hyp_k, s1 - s2)
