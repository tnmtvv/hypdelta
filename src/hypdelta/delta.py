from hypdelta.naive import delta_naive_cpu, delta_naive_gpu
from hypdelta.fast import delta_CCL_cpu, delta_CCL_gpu
from hypdelta.condensed import delta_condensed
from hypdelta.cartesian import delta_cartesian


class GPUNotImplemented(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Not implemented for GPU, use CPU version instead"):
        self.message = message
        super().__init__(self.message)


class CPUNotImplemented(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Not implemented for CPU, use GPU version instead"):
        self.message = message
        super().__init__(self.message)


def hypdelta(
    distance_matrix,
    device=["cpu", "gpu"],
    strategy=["naive", "condensed", "heuristic", "CCL", "cartesian"],
    l=0.05,
    tries=25,
    heuristic=True,
    threadsperblock=(16, 16, 4),
    max_threads=1024,
    max_gpu_mem=16,
):
    """
    Computes the delta hyperbolicity of a distance matrix using various strategies and devices.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        The distance matrix for which delta hyperbolicity is to be computed.

    device : list of str, optional
        The device to use for computation, can be "cpu" or "gpu". Default is ["cpu", "gpu"].

    strategy : list of str, optional
        The strategy to use for computation. Options are "naive", "condensed", "heuristic", "CCL", and "cartesian". Default is ["naive", "condensed", "heuristic", "CCL", "cartesian"].

    l : float, optional
        A parameter for certain strategies like "CCL". Default is 0.05.

    tries : int, optional
        Number of tries for the "condensed" strategy. Default is 25.

    heuristic : bool, optional
        Whether to use heuristic methods for the "condensed" strategy. Default is True.

    threadsperblock : tuple of int, optional
        The number of threads per block for GPU computation. Default is (16, 16, 4).

    max_threads : int, optional
        The maximum number of threads to use for GPU computation in the "cartesian" strategy. Default is 1024.

    max_gpu_mem : int, optional
        The maximum gpu memory in Gb. Used in "cartesian" strategy. Default is 16.

    Returns:
    --------
    float
        The computed delta hyperbolicity of the distance matrix.

    Raises:
    -------
    GPUNotImplemented
        If the chosen strategy is not implemented for the GPU device.

    Notes:
    ------
    The function determines the delta hyperbolicity using the specified strategy and device. Different strategies have different computational complexities and are suitable for different types of data and computational resources.
    """
    if strategy == "naive":
        if device == "cpu":
            delta = delta_naive_cpu(dist_matrix=distance_matrix)
        elif device == "gpu":
            delta = delta_naive_gpu(
                dist_matrix=distance_matrix, threadsperblock=threadsperblock
            )
    elif strategy == "condensed":
        if device == "cpu":
            delta = delta_condensed(
                dist_matrix=distance_matrix, tries=tries, heuristic=heuristic
            )
        elif device == "gpu":
            raise GPUNotImplemented(
                "The 'condensed' strategy is not implemented for GPU."
            )
    elif strategy == "CCL":
        if device == "cpu":
            delta = delta_CCL_cpu(dist_matrix=distance_matrix, l=l)
        elif device == "gpu":
            delta = delta_CCL_gpu(
                dist_matrix=distance_matrix, l=l, threadsperblock=threadsperblock
            )
    elif strategy == "cartesian":
        delta = delta_cartesian(
            dist_matrix=distance_matrix,
            l=l,
            max_threads=max_threads,
            max_gpu_mem=max_gpu_mem,
        )
    return delta
