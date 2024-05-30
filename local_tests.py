import pytest

from utils import *
from hyppy import hypdelta
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
def test_CCL_CPU_GPU():
    dist_matrix = generate_dists(500)

    delta_GPU = hypdelta(
        dist_matrix,
        device="gpu",
        strategy="CCL",
        l=0.05,
    )

    delta_CPU = hypdelta(
        dist_matrix,
        device="cpu",
        strategy="CCL",
        l=0.05,
    )

    assert delta_GPU == pytest.approx(delta_CPU, 0.001)


@pytest.mark.parametrize(
    "points",
    [
        generate_synthetic_points(100, 100),
        generate_synthetic_points(200, 200),
        generate_synthetic_points(500, 500),
        generate_synthetic_points(500, 1000),
    ],
)
def test_CCL_batch(points):

    dist_matrix = pairwise_distances(points)

    delta_CCL = hypdelta(
        dist_matrix,
        device="gpu",
        strategy="CCL",
        l=0.05,
    )

    delta_cartesian = hypdelta(
        dist_matrix,
        device="gpu",
        strategy="cartesian",
        l=0.05,
    )

    assert delta_CCL == pytest.approx(delta_cartesian, 0.001)
