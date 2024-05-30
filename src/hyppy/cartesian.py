import numpy as np
from numba import cuda

from hyppy.calculus_utils import (
    get_far_away_pairs,
    prepare_batch_indices_flat,
    batch_flatten,
    calc_max_lines,
)
from hyppy.cudaprep import cuda_prep_cartesian


@cuda.jit
def gpu_cartesian(dist_array, delta_res):
    """
    Computes the delta hyperbolicity for a given array of distance values using Cartesian coordinates.

    This function is designed to run on a GPU using CUDA. It calculates the delta hyperbolicity value
    for each row in the distance array and updates the result atomically.

    Parameters:
    -----------
    dist_array : np.ndarray
        A 2D array where each row contains six distance values corresponding to the distances
        between pairs of points in the Cartesian space.
    delta_res : np.ndarray
        A single-element array to store the maximum delta hyperbolicity value computed. This array
        is updated atomically to ensure thread safety.

    Notes:
    ------
    - The function uses CUDA for parallel computation, so it must be run in an environment with a
      compatible GPU and the necessary CUDA setup.
    - The distance array should be prepared such that each row contains exactly six values.
    - The delta hyperbolicity value is computed as:
      (d(A,B) + d(C,D) - max(d(A,C) + d(B,D), d(A,D) + d(B,C))) / 2
    - This function should be launched as a CUDA kernel with an appropriate number of threads and blocks.
    """

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


def delta_cartesian(dist_matrix, l, all_threads):
    """
    Computes the delta hyperbolicity of a given distance matrix using a Cartesian approach.

    This function uses a batching technique to handle large computations efficiently and leverages
    GPU processing for performance improvements.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        A 2D array representing the distance matrix.
    l : int
        A parameter to determine the fraction of far away pairs to consider.
    all_threads : int
        The number of threads to use for GPU computation.

    Returns:
    --------
    delta : float
        The maximum delta hyperbolicity value computed.
    diam : float
        The diameter of the distance matrix.

    Notes:
    ------
    - The function assumes that CUDA and the necessary GPU setup are available and correctly configured.
    - The distance matrix should be a square matrix representing the pairwise distances between points.
    """
    n = dist_matrix.shape[0]
    diam = np.max(dist_matrix)

    far_away_pairs = get_far_away_pairs(dist_matrix, int((n * (n + 1) / 2) * l))
    len_far_away = len(far_away_pairs)

    batch_size = calc_max_lines(self.mem_gpu_bound, len_far_away)

    cartesian_size = int(len_far_away * (len_far_away - 1) / 2)
    batch_N = np.ceil(cartesian_size // batch_size)

    deltas = np.empty(batch_N)

    for i in range(batch_N):
        (indices) = prepare_batch_indices_flat(
            far_away_pairs,
            i * batch_size,
            min((i + 1) * batch_size, cartesian_size),
            n,
        )

        batch = batch_flatten(indices.ravel(), dist_matrix.ravel()).reshape(-1, 6)
        (
            cartesian_dist_array,
            delta_res,
            threadsperblock,
            blockspergrid,
        ) = cuda_prep_cartesian(batch, all_threads)
        gpu_cartesian[blockspergrid, threadsperblock](cartesian_dist_array, delta_res)
        deltas[i] = 2 * delta_res[0] / diam
        del cartesian_dist_array
        del batch
    delta = max(deltas)
    return delta, diam
