import math
from numba import cuda
import numpy as np


def cuda_prep_true_delta(dist_matrix):
    k_1 = int(math.log2(dist_matrix.shape[0])) + 1
    k_2 = 2**k_1 - 1
    k_3 = dist_matrix.shape[0]
    adj_m = cuda.to_device(dist_matrix.flatten())

    k = cuda.to_device([k_2, k_1, k_3])
    delta_res = cuda.to_device(np.zeros(1))

    print([k_2, k_1, k_3])

    block_size = (16, 16, 4)
    threadsperblock = block_size
    blockspergrid_x = min(
        65535, int(np.ceil(dist_matrix.shape[0] ** 2 / threadsperblock[0])) + 1
    )
    blockspergrid_y = min(
        65535, int(np.ceil(dist_matrix.shape[0] / threadsperblock[1])) + 1
    )
    blockspergrid_z = min(
        65535, int(np.ceil(dist_matrix.shape[0] / threadsperblock[2])) + 1
    )

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    return adj_m, k, delta_res, threadsperblock, blockspergrid


def cuda_prep_CCL(far_away_pairs, dist_matrix, block_size):
    x_coords, y_coords = (
        list(zip(*far_away_pairs))[0],
        list(zip(*far_away_pairs))[1],
    )
    x_coords = np.asarray(x_coords).astype(int)
    y_coords = np.asarray(y_coords).astype(int)

    x_coord_pairs = cuda.to_device(x_coords)
    y_coord_pairs = cuda.to_device(y_coords)
    # dist_matrix = matrix_to_triangular(dist_matrix)
    adj_m = cuda.to_device(dist_matrix)
    # results = cuda.to_device(list(np.zeros(len(x_coord_pairs))))
    delta_res = cuda.to_device(list(np.zeros(1)))
    n = len(x_coord_pairs)

    threadsperblock = (block_size, block_size)
    blockspergrid_x = int(np.ceil(n / threadsperblock[0])) + 1
    blockspergrid_y = int(np.ceil(n / threadsperblock[1])) + 1
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return (
        n,
        x_coord_pairs,
        y_coord_pairs,
        adj_m,
        # results,
        blockspergrid,
        threadsperblock,
        delta_res,
    )


def cuda_prep_cartesian(dist_array, block_size):
    threadsperblock = block_size

    blockspergrid = min(65535, int(dist_array.shape[0] / threadsperblock) + 1)
    cartesian_dist_array = cuda.to_device(np.asarray(dist_array))
    delta_res = cuda.to_device(np.zeros(1))
    return cartesian_dist_array, delta_res, threadsperblock, blockspergrid
