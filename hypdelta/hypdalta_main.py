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
    strategy=["naive", "condensed", "heuristic", "CCL"],
    l=0.05,
    tries=25,
    heuristic=True,
    threadsperblock=(16, 16, 4),
):
    """
    Compute the delta-hyperbolicity of a distance matrix using various strategies on different devices.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        The matrix containing pairwise distances between points.
    device : list of str, optional
        The device(s) to run the computation on, either "cpu" or "cuda". Default is ["cpu", "cuda"].
    strategy : list of str, optional
        The strategy to use for computation. Options are "naive", "condenced", "heuristic", and "CCL". Default is ["naive", "condenced", "heuristic", "CCL"].
    l : float, optional
        The proportion of far-away pairs to consider for the CCL strategy. Default is 0.05.
    tries : int, optional
        The number of tries for the "condensed" strategy. Default is 25.
    heuristic : bool, optional
        Whether to use heuristics for the "condenced" strategy. Default is True.
    threadsperblock : tuple of int, optional
        Parameter needed for "CCL" and "naive" strategies when device == "gpu".
        The number of threads per block for GPU computation. (Tuple of 2 for CCL on gpu strategy, tuple of 3 for naive strategy). Default is (16, 16, 4).

    Returns
    -------
    float
        The computed delta-hyperbolicity.

    Raises
    ------
    GPUNotImplemented
        If the selected strategy is not implemented for GPU.

    Notes
    -----
    This function supports multiple strategies and devices for computing the delta-hyperbolicity of a distance matrix:
    - "naive" strategy is implemented for both CPU and GPU.
    - "condenced" strategy is implemented for CPU only.
    - "CCL" strategy is implemented for both CPU and GPU.
    """
    if strategy == "naive":
        if device == "cpu":
            delta = delta_naive_cpu(distance_matrix)
        elif device == "gpu":
            delta = delta_naive_gpu(distance_matrix, threadsperblock)
    elif strategy == "condensed":
        if device == "cpu":
            delta = delta(distance_matrix, tries, heuristic)
        elif device == "gpu":
            raise GPUNotImplemented(
                "The 'condensed' strategy is not implemented for GPU."
            )
    if strategy == "CCL":
        if device == "cpu":
            delta = delta_CCL_cpu(distance_matrix, l)
        elif device == "gpu":
            delta = delta_CCL_gpu(distance_matrix, l, threadsperblock)
    if strategy == "cartesian":
        delta = delta_cartesian()
    return delta
