from hypdelta import hypdelta
import numpy as np


def generate_synthetic_points(dimensions, num_points):
    points = np.random.rand(num_points, dimensions)
    return points


def distance(point1, point2):
    diff_squared = np.square(point1 - point2)
    sum_diff_squared = np.sum(diff_squared)
    dist = np.sqrt(sum_diff_squared)
    return dist


def build_dist_matrix(data):
    arr_all_dist = []
    for p in data:
        arr_dist = list(map(lambda x: 0 if (p == x).all() else distance(p, x), data))
        arr_all_dist.append(arr_dist)
    arr_all_dist = np.asarray(arr_all_dist)
    return arr_all_dist


def generate_dists(dim=100):
    # dim = 100
    points = generate_synthetic_points(dim, dim)
    dist_arr = build_dist_matrix(points)
    return dist_arr


def get_far_away_pairs(A, N):
    a = zip(*np.unravel_index(np.argsort(-A.ravel())[:N], A.shape))
    return [(i, j) for (i, j) in a if i < j]
