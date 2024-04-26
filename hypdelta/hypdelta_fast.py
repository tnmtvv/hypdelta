import numpy as np
from numba import njit, prange, typed, cuda

from hypdelta.utils import get_far_away_pairs, prepare_batch_indices_flat
from hypdelta.cudaprep import cuda_prep_CCL, cuda_prep_cartesian


def CCL_cpu(dist_matrix, **kwargs):
    diam = np.max(dist_matrix)
    far_away_pairs = get_far_away_pairs(
        dist_matrix, int(dist_matrix.shape[0] * dist_matrix.shape[0] * kwargs["l"])
    )
    max_iter = 100000
    delta = delta_hyp_CCL(typed.List(far_away_pairs), dist_matrix)
    return 2 * delta / diam, diam


def CCL_gpu(dist_matrix, **kwargs):
    diam = np.max(dist_matrix)
    far_away_pairs = get_far_away_pairs(
        dist_matrix, dist_matrix.shape[0] * dist_matrix.shape[0] * kwargs["l"]
    )
    (
        n,
        x_coord_pairs,
        y_coord_pairs,
        adj_m,
        blockspergrid,
        threadsperblock,
        delta_res,
    ) = cuda_prep_CCL(far_away_pairs, dist_matrix, 32)
    delta_hyp_CCL_GPU[blockspergrid, threadsperblock](
        n, x_coord_pairs, y_coord_pairs, adj_m, delta_res
    )
    delta, _ = 2 * delta_res[0] / diam
    return delta, diam


@njit(parallel=True, fastmath=True)
def delta_hyp_CCL(far_apart_pairs, adj_m: np.ndarray):
    """
    Computes Gromov's delta-hyperbolicity value with the basic approach, proposed in the article
    "On computing the Gromov hyperbolicity", 2015, by Nathann Cohen, David Coudert, Aurélien Lancin.

    Parameters:
    -----------
    far_apart_pairs : numpy.ndarray
        List of pairs of points, sorted by decrease of distance i.e. the most distant pair must be the first one.

    adj_m: numpy.ndarry
        Distance matrix.

    Returns:
    --------
    float
        The delta hyperbolicity value.
    """
    delta_hyp = 0.0
    for iter_1 in prange(1, len(far_apart_pairs)):
        pair_1 = far_apart_pairs[iter_1]
        for iter_2 in prange(iter_1):
            pair_2 = far_apart_pairs[iter_2]
            i = pair_1[0]
            j = pair_1[1]
            v = pair_2[0]
            w = pair_2[1]

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
def delta_hyp_CCL_GPU(n, fisrt_points, second_points, adj_m, delta_res):
    """
    Computes Gromov's delta-hyperbolicity value with the basic approach, proposed in the article
    "On computing the Gromov hyperbolicity", 2015, by Nathann Cohen, David Coudert, Aurélien Lancin.
    Algorithm was rewritten for execution on GPU.

    Parameters:
    -----------
    n: int
        The number of pairs.

    pairs_x_coord:
        List of the fisrt points of the far away pairs pairs.


    pairs_y_coord:
        List of the second points of the far away pairs pairs.


    adj_m: numpy.ndarry
        Distance matrix.

    x_coords_pairs
    far_apart_pairs: numpy.ndarray
        List of pairs of points, sorted by decrease of distance i.e. the most distant pair must be the first one.

    adj_m: numpy.ndarry
        Distance matrix.

    results:
        Array, where deltas for each pair will be stored.

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
                delta_res, (0), (d_ij + d_vw - max(d_iv + d_jw, d_iw + d_jv)) / 2
            )
