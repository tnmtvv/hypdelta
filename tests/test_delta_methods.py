import pytest

from utils import *
from hyppy import hypdelta
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist


def test_CCL_true_delta():
    points = generate_synthetic_points(100, 100)

    dist_matrix = pairwise_distances(points)

    delta_CCL = hypdelta(
        dist_matrix,
        device="cpu",
        strategy="CCL",
        l=0.05,
    )

    delta_naive = hypdelta(dist_matrix, device="cpu", strategy="naive")

    assert delta_CCL == pytest.approx(delta_naive, 0.001)


# def test_CCL_GPU():
#     dist_matrix = generate_dists(500)

#     strategy_CCL = CCLStrategy()
#     strategy_gpu = GPUStrategy()

#     delta_GPU = strategy_gpu.calculate_delta(dist_matrix)
#     delta_CCL = strategy_CCL.calculate_delta(dist_matrix)

#     assert delta_GPU == pytest.approx(delta_CCL, 0.001)


# def test_GPU_true_delta():
#     dist_matrix = generate_dists(500)

#     strategy_CCL = GPUStrategy()
#     strategy_true = TrueDeltaGPUStrategy()

#     delta_GPU = strategy_CCL.calculate_delta(dist_matrix)
#     delta_true = strategy_true.calculate_delta(dist_matrix)

#     assert delta_GPU == pytest.approx(delta_true, 0.001)


def test_condenced_true_delta():
    points = generate_synthetic_points(100, 100)

    dist_matrix = pairwise_distances(points)
    dist_matrix_condesed = pdist(points)

    delta_condensed = hypdelta(
        dist_matrix_condesed,
        device="cpu",
        strategy="condensed",
        tries=25,
        heuristic=False,
    )

    delta_condensed_heuristic = hypdelta(
        dist_matrix, device="cpu", strategy="condensed", tries=25
    )

    assert delta_condensed == pytest.approx(delta_condensed_heuristic, 0.001)
