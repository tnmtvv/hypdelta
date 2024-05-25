from hypdelta.hypdelta_naive import naive_cpu, naive_gpu
from hypdelta.hypdelta_fast import CCL_cpu, CCL_gpu
from hypdelta.hypdelta_condensed import calculate_condenced_delta


class GPUNotImplemened(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Not implemented for GPU, use CPU version instead"):
        self.message = message
        super().__init__(self.message)


def hypdelta(
    distance_matrix,
    device=["cpu", "cuda"],
    strategy=["naive", "condenced", "heuristic", "CCL"],
    **kwargs
):  # device can be cuda
    if strategy == "naive":
        if device == "cpu":
            print(kwargs)
            delta = naive_cpu(distance_matrix, **kwargs)
        elif device == "gpu":
            delta = naive_gpu(distance_matrix, **kwargs)
    elif strategy == "condenced":
        if device == "cpu":
            delta = calculate_condenced_delta(
                distance_matrix, **kwargs
            )  # heuristic=True
        elif device == "gpu":
            raise GPUNotImplemened
    if strategy == "CCL":
        if device == "cpu":
            delta = CCL_cpu(distance_matrix, **kwargs)
        elif device == "gpu":
            delta = CCL_gpu(distance_matrix, **kwargs)
    return delta
