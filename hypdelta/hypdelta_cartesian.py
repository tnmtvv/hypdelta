import numpy as np
from numba import cuda

from hypdelta.utils import get_far_away_pairs, prepare_batch_indices_flat, batch_flatten
from hypdelta.cudaprep import cuda_prep_CCL, cuda_prep_cartesian


def delta_cartesian_way_new(dist_matrix, batch_size, l):
    far_away_pairs = get_far_away_pairs(
        dist_matrix, dist_matrix.shape[0] * dist_matrix.shape[0] * l
    )
    print("new way")
    diam = np.max(dist_matrix)
    print(f"diam: {diam}")
    cartesian_size = int(len(far_away_pairs) * (len(far_away_pairs) - 1) / 2)
    batch_N = int(cartesian_size // batch_size) + 1
    deltas = np.empty(batch_N)
    print(f"shape X: {dist_matrix.shape}")
    print(f"shape far_away_pairs: {far_away_pairs.shape}")
    print(f"all_size: {cartesian_size}")
    print(f"batch_size: {batch_size}")
    print(f"batches: {batch_N}")
    # times = []
    for i in range(batch_N):
        print(f"{i} batch started")
        (indices) = prepare_batch_indices_flat(
            far_away_pairs,
            i * batch_size,
            min((i + 1) * batch_size, cartesian_size),
            dist_matrix.shape,
        )
        print(indices.ravel().shape)
        indices = prepare_batch_indices_flat(
            far_away_pairs, i * batch_size, min((i + 1) * batch_size, cartesian_size)
        )

        batch = batch_flatten(indices.ravel(), dist_matrix.ravel()).reshape(-1, 6)
        # batch_time = timer() - batch_time_start
        print("batch built")
        # times.append(batch_time)
        (
            cartesian_dist_array,
            delta_res,
            threadsperblock,
            blockspergrid,
        ) = cuda_prep_cartesian(batch, 1024)
        delta_CCL_cartesian[blockspergrid, threadsperblock](
            cartesian_dist_array, delta_res
        )
        deltas[i] = 2 * delta_res[0] / diam
        del cartesian_dist_array
        del batch
    delta = max(deltas)
    return delta, diam


@cuda.jit
def delta_CCL_cartesian(dist_array, delta_res):
    row = cuda.grid(1)
    cuda.atomic.max(
        delta_res,
        (0),
        (
            dist_array[row][0]
            + dist_array[row][1]
            - max(
                dist_array[row][2] + dist_array[row][3],
                dist_array[row][4] + dist_array[row][5],
            )
        )
        / 2,
    )
