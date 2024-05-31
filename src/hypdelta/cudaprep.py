import math
from numba import cuda
import numpy as np
from typing import Tuple


def cuda_prep_naive(
    dist_matrix: np.ndarray, threadsperblock: Tuple[int, int, int]
) -> Tuple[
    cuda.devicearray.DeviceNDArray,
    cuda.devicearray.DeviceNDArray,
    cuda.devicearray.DeviceNDArray,
    cuda.devicearray.DeviceNDArray,
    Tuple[int, int, int],
    Tuple[int, int, int],
]:
    """
    Prepares the necessary CUDA variables for the naive delta hyperbolicity calculation.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        The distance matrix.
    threadsperblock : Tuple[int, int, int]
        The number of threads per block.

    Returns:
    --------
    Tuple containing:
        - adj_m : cuda.devicearray.DeviceNDArray
            The distance matrix flattened and transferred to the device.
        - k : cuda.devicearray.DeviceNDArray
            An array containing parameters for the CUDA kernel.
        - diff_num_arr : cuda.devicearray.DeviceNDArray
            An array containing the difference between the nearest power of two greater than the needed number of threads
            and the nearest power of two greater than the actual number of available threads.
        - delta_res : cuda.devicearray.DeviceNDArray
            An array to store the result of the delta calculation.
        - threadsperblock : Tuple[int, int, int]
            The number of threads per block.
        - blockspergrid : Tuple[int, int, int]
            The number of blocks per grid.
    """

    # Calculate the log base 2 of the number of rows in the distance matrix, plus one
    k_1 = int(math.log2(dist_matrix.shape[0])) + 1

    # Calculate the nearest power of two greater than the number of rows in the distance matrix
    k_2 = 2**k_1 - 1

    # Get the number of rows in the distance matrix
    k_3 = dist_matrix.shape[0]

    # Transfer the flattened distance matrix to the GPU
    adj_m = cuda.to_device(dist_matrix.flatten())

    # Transfer the k parameters to the GPU
    k = cuda.to_device([k_2, k_1, k_3])

    # Allocate memory on the GPU to store the result of the delta calculation
    delta_res = cuda.to_device(np.zeros(1))

    # Calculate the difference between the nearest power of two greater than the needed number of threads
    # and the nearest power of two greater than the actual number of available threads
    diff_num = int(
        (2 * k_1)
        - math.log2(min(65536, int((2 ** (2 * k_1)) / threadsperblock[0])) * 16)
    )

    # Transfer the diff_num to the GPU
    diff_num_arr = cuda.to_device([diff_num])

    # Calculate the number of blocks per grid for each dimension, making sure it doesn't exceed 65535
    blockspergrid_x = min(65535, int((2 ** (2 * k_1)) / threadsperblock[0]) - 1)
    blockspergrid_y = min(
        65535, int(np.ceil(dist_matrix.shape[0] / threadsperblock[1])) + 1
    )
    blockspergrid_z = min(
        65535, int(np.ceil(dist_matrix.shape[0] / threadsperblock[2])) + 1
    )

    # Combine the number of blocks per grid into a tuple
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Return all prepared variables
    return adj_m, k, diff_num_arr, delta_res, threadsperblock, blockspergrid


def cuda_prep_CCL(
    far_away_pairs: list, dist_matrix: np.ndarray, threadsperblock: Tuple[int, int]
) -> Tuple[
    int,
    cuda.devicearray.DeviceNDArray,
    cuda.devicearray.DeviceNDArray,
    cuda.devicearray.DeviceNDArray,
    Tuple[int, int],
    Tuple[int, int],
    cuda.devicearray.DeviceNDArray,
]:
    """
    Prepares the necessary CUDA variables for the CCL delta hyperbolicity calculation.

    Parameters:
    -----------
    far_away_pairs : list of tuples
        List of far away pairs, where each pair is a tuple of coordinates.
    dist_matrix : np.ndarray
        The distance matrix.
    threadsperblock : Tuple[int, int]
        The block size for CUDA kernels.

    Returns:
    --------
    Tuple containing:
        - n : int
            The number of far away pairs.
        - x_coord_pairs : cuda.devicearray.DeviceNDArray
            X coordinates of the far away pairs transferred to the device.
        - y_coord_pairs : cuda.devicearray.DeviceNDArray
            Y coordinates of the far away pairs transferred to the device.
        - adj_m : cuda.devicearray.DeviceNDArray
            The distance matrix transferred to the device.
        - blockspergrid : Tuple[int, int]
            The number of blocks per grid.
        - threadsperblock : Tuple[int, int]
            The number of threads per block.
        - delta_res : cuda.devicearray.DeviceNDArray
            An array to store the result of the delta calculation.
    """
    # Extract x and y coordinates from the far away pairs and convert them to numpy arrays
    x_coords, y_coords = map(np.asarray, zip(*far_away_pairs))

    # Transfer the x and y coordinates to the device (GPU)
    x_coord_pairs = cuda.to_device(x_coords.astype(int))
    y_coord_pairs = cuda.to_device(y_coords.astype(int))

    # Transfer the distance matrix to the device (GPU)
    adj_m = cuda.to_device(dist_matrix)

    # Create a device array to store the delta result and transfer it to the device
    delta_res = cuda.to_device(list(np.zeros(1)))

    # The number of far away pairs
    n = len(x_coord_pairs)

    # Calculate the number of blocks per grid for x and y dimensions
    blockspergrid_x = int(np.ceil(n / threadsperblock[0])) + 1
    blockspergrid_y = int(np.ceil(n / threadsperblock[1])) + 1
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Return the prepared CUDA variables and configuration
    return (
        n,
        x_coord_pairs,
        y_coord_pairs,
        adj_m,
        blockspergrid,
        threadsperblock,
        delta_res,
    )


def cuda_prep_cartesian(
    dist_array: np.ndarray, threadsperblock: int
) -> Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, int, int]:
    """
    Prepares the necessary CUDA variables for the Cartesian delta hyperbolicity calculation.

    Parameters:
    -----------
    dist_array : np.ndarray
        The distance array for the Cartesian product.
    threadsperblock : int
        The block size for CUDA kernels.

    Returns:
    --------
    Tuple containing:
        - cartesian_dist_array : cuda.devicearray.DeviceNDArray
            The distance array transferred to the device.
        - delta_res : cuda.devicearray.DeviceNDArray
            An array to store the result of the delta calculation.
        - threadsperblock : int
            The number of threads per block.
        - blockspergrid : int
            The number of blocks per grid.
    """

    blockspergrid = min(65535, int(dist_array.shape[0] / threadsperblock) + 1)
    cartesian_dist_array = cuda.to_device(np.asarray(dist_array))
    delta_res = cuda.to_device(np.zeros(1))
    return cartesian_dist_array, delta_res, threadsperblock, blockspergrid
