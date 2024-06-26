import pytest

from tests.utils import *

from src.hypdelta.delta import hypdelta
from sklearn.metrics import pairwise_distances


@pytest.mark.parametrize(
    "points",
    [
        generate_synthetic_points(100, 100),
        generate_synthetic_points(200, 200),
        generate_synthetic_points(500, 500),
    ],
)
def test_CCL_true_delta(points):

    dist_matrix = pairwise_distances(points)

    delta_CCL = hypdelta(
        dist_matrix,
        device="cpu",
        strategy="CCL",
        l=0.5,
    )

    delta_naive = hypdelta(dist_matrix, device="cpu", strategy="naive")

    assert delta_CCL == pytest.approx(delta_naive, 0.01)
