import numpy as np


def generate_synthetic_points(dimensions, num_points):
    """
    Generates a set of synthetic points in a given dimensional space.

    Parameters:
    -----------
    dimensions : int
        The number of dimensions for each point.
    num_points : int
        The number of points to generate.

    Returns:
    --------
    np.ndarray
        An array of shape (num_points, dimensions) containing the generated points.
    """
    points = np.random.rand(num_points, dimensions)
    return points


def build_dist_matrix(data):
    """
    Builds a distance matrix for a given set of points.

    Parameters:
    -----------
    data : np.ndarray
        An array of shape (num_points, dimensions) containing the points.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (num_points, num_points) where each element [i, j] is the Euclidean distance between points i and j.
    """
    arr_all_dist = []
    for p in data:
        arr_dist = list(
            map(lambda x: 0 if (p == x).all() else np.linalg.norm(p - x), data)
        )
        arr_all_dist.append(arr_dist)
    arr_all_dist = np.asarray(arr_all_dist)
    return arr_all_dist


def generate_dists(dim=100, num_points=100):
    """
    Generates a distance matrix for a set of synthetic points in a given dimensional space.

    Parameters:
    -----------
    dim : int, optional
        The number of dimensions for each point (default is 100).
    num_points : int, optional
        The number of points to generate (default is 100).

    Returns:
    --------
    np.ndarray
        A 2D array of shape (num_points, num_points) containing the distances between each pair of points.
    """
    points = generate_synthetic_points(dim, num_points)
    dist_arr = build_dist_matrix(points)
    return dist_arr


def get_far_away_pairs(A, N):
    """
    Finds the farthest N pairs of points from a distance matrix.

    Parameters:
    -----------
    A : np.ndarray
        A 2D array where each element [i, j] is the distance between points i and j.
    N : int
        The number of far away pairs to find.

    Returns:
    --------
    list of tuples
        A list of tuples, each containing a pair of indices (i, j) corresponding to the farthest pairs.
    """
    a = zip(*np.unravel_index(np.argsort(-A.ravel())[:N], A.shape))
    return [(i, j) for (i, j) in a if i < j]
