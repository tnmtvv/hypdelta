# hypdelta
# ![Logo](logo.svg)

![example workflow](https://github.com/tnmtvv/hypdelta/actions/workflows/tests.yml/badge.svg)

`hypdelta` is a Python library for calculating delta hyperbolicity of distance matrices using various strategies and computational devices (CPU/GPU). It provides flexibility in choosing the method and device for computation to balance between accuracy and performance.

## Features

- **Multiple Strategies**: Supports naive, condensed, heuristic, CCL, and cartesian strategies for calculating delta hyperbolicity.
- **Device Flexibility**: Can run on both CPU and GPU.
- **Customizable Parameters**: Allows for setting parameters like block size, number of tries, and heuristic options.

## Installation

To install `hypdelta`, you can clone the repository and install the requirements:

```bash
git clone https://github.com/tnmtvv/hypdelta.git
cd hypdelta
pip install -r requirements.txt
```

## Usage

Here's a basic example to get you started with `hypdelta`:

```python
import numpy as np
from hypdelta import hypdelta

# Generate a synthetic distance matrix
def generate_synthetic_points(dimensions, num_points):
    points = np.random.rand(num_points, dimensions)
    return points

def build_dist_matrix(data):
    arr_all_dist = []
    for p in data:
        arr_dist = list(
            map(lambda x: 0 if (p == x).all() else np.linalg.norm(p - x), data)
        )
        arr_all_dist.append(arr_dist)
    arr_all_dist = np.asarray(arr_all_dist)
    return arr_all_dist

def generate_dists(dim=100, num_points=100):
    points = generate_synthetic_points(dim, num_points)
    dist_arr = build_dist_matrix(points)
    return dist_arr

distance_matrix = generate_dists(dim=10, num_points=50)

# Calculate delta hyperbolicity using the naive strategy on CPU
delta = hypdelta(distance_matrix, device="cpu", strategy="naive")
print(f"Delta hyperbolicity (naive, CPU): {delta}")

# Calculate delta hyperbolicity using the CCL strategy on GPU
delta = hypdelta(distance_matrix, device="gpu", strategy="CCL", l=0.1)
print(f"Delta hyperbolicity (CCL, GPU): {delta}")
```

### Strategies and Devices

The `hypdelta` function supports the following strategies:

- `"naive"`: A straightforward approach to calculate delta hyperbolicity.
- `"condensed"`: A strategy that uses condensed data representation.
- `"heuristic"`: A heuristic-based approach for faster computation.
- `"CCL"`: A strategy using far-away pairs for computation.
- `"cartesian"`: A strategy that utilizes the cartesian product of pairs.

And the following devices:

- `"cpu"`: Computation on the CPU.
- `"gpu"`: Computation on the GPU.

### Parameters

- `distance_matrix`: The distance matrix for which delta hyperbolicity is to be computed.
- `device`: The device to use for computation, can be `"cpu"` or `"gpu"`.
- `strategy`: The strategy to use for computation. Options are `"naive"`, `"condensed"`, `"heuristic"`, `"CCL"`, and `"cartesian"`.
- `l`: A parameter for certain strategies like `"CCL"`. Default is 0.05.
- `tries`: Number of tries for the `"condensed"` strategy. Default is 25.
- `heuristic`: Whether to use heuristic methods for the `"condensed"` strategy. Default is True.
- `threadsperblock`: The number of threads per block for GPU computation. Default is (16, 16, 4).
- `max_threads`: The maximum number of threads to use for GPU computation in the `"cartesian"` strategy. Default is 1024.
- `max_gpu_mem` : The maximum gpu memory in Gb. Used in `"cartesian"` strategy. Default is 16.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


---

Feel free to explore the repository and experiment with different strategies and devices to find the optimal settings for your use case.