import pytest

from tests.utils import *

from src.hyppy import hypdelta
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist


@pytest.mark.parametrize(
    "points",
    [
        generate_synthetic_points(100, 100),
        generate_synthetic_points(200, 200),
        generate_synthetic_points(500, 500),
        generate_synthetic_points(500, 1000),
    ],
)
def test_CCL_true_delta(points):

    dist_matrix = pairwise_distances(points)

    delta_CCL = hypdelta(
        dist_matrix,
        device="cpu",
        strategy="CCL",
        l=0.1,
    )

    delta_naive = hypdelta(dist_matrix, device="cpu", strategy="naive")

    assert delta_CCL == pytest.approx(delta_naive, 0.001)


@pytest.mark.parametrize(
    "points",
    [
        generate_synthetic_points(100, 100),
        generate_synthetic_points(200, 200),
        generate_synthetic_points(500, 500),
        generate_synthetic_points(500, 1000),
    ],
)
def test_condenced_true_delta(points):

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
