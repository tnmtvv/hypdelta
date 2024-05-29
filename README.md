# HypDelta

HypDelta is a tool for calculating the delta hyperbolicity of distance matrices using various strategies. It leverages both CPU and GPU computations to provide efficient and scalable performance.

## Features

- **Multiple Calculation Strategies:** Supports `naive`, `condensed`, `heuristic`, and `CCL` strategies for calculating delta hyperbolicity.
- **GPU Acceleration:** Utilizes CUDA for accelerated computations on supported hardware.
- **Flexible Configuration:** Allows customization of computational parameters, such as threads per block for GPU calculations.

## Installation

To use HypDelta, clone the repository and install the required dependencies:

```bash
git clone https://github.com/tnmtvv/hypdelta.git
cd hypdelta
pip install -r requirements.txt
```

## Usage

### Generating Synthetic Points

You can generate synthetic points in a given dimension and build a distance matrix from them:

```python
import numpy as np
from hypdelta import generate_synthetic_points, build_dist_matrix

# Generate synthetic points
dimensions = 100
num_points = 100
points = generate_synthetic_points(dimensions, num_points)

# Build distance matrix
dist_matrix = build_dist_matrix(points)
```

### Calculating Delta Hyperbolicity

You can calculate the delta hyperbolicity using different strategies. Here's an example using the `naive` strategy on the CPU:

```python
from hypdelta import hypdelta

# Parameters
device = "cpu"
strategy = "naive"
threadsperblock = (16, 16, 4)

# Calculate delta hyperbolicity
delta, diam = hypdelta(dist_matrix, device=device, strategy=strategy, threadsperblock=threadsperblock)

print(f"Delta: {delta}, Diameter: {diam}")
```

You can also use GPU for calculations if supported:

```python
device = "cuda"

# Calculate delta hyperbolicity on GPU
delta, diam = hypdelta(dist_matrix, device=device, strategy=strategy, threadsperblock=threadsperblock)

print(f"Delta: {delta}, Diameter: {diam}")
```

## Functions

### `generate_synthetic_points(dimensions, num_points)`

Generates synthetic points in a given dimension.

**Parameters:**
- `dimensions` (int): The number of dimensions for each point.
- `num_points` (int): The number of points to generate.

**Returns:**
- `np.ndarray`: An array of shape `(num_points, dimensions)` containing the generated points.

### `build_dist_matrix(data)`

Builds a distance matrix from a given set of points.

**Parameters:**
- `data` (np.ndarray): An array of points.

**Returns:**
- `np.ndarray`: A distance matrix.

### `hypdelta(distance_matrix, device, strategy, l=0.05, tries=25, heuristic=True, threadsperblock=(16, 16, 4))`

Calculates delta hyperbolicity using the specified strategy.

**Parameters:**
- `distance_matrix` (np.ndarray): The distance matrix.
- `device` (str): The device to use for computation (`"cpu"` or `"cuda"`).
- `strategy` (str): The calculation strategy (`"naive"`, `"condensed"`, `"heuristic"`, `"CCL"`).
- `l` (float, optional): Parameter for CCL strategy. Default is `0.05`.
- `tries` (int, optional): Number of tries for condensed strategy. Default is `25`.
- `heuristic` (bool, optional): Use heuristic for condensed strategy. Default is `True`.
- `threadsperblock` (tuple, optional): Number of threads per block for GPU computation. Default is `(16, 16, 4)`.

**Returns:**
- `tuple`: A tuple containing the delta and the diameter.

## Examples

### Generating and Using Distance Matrices

Here is a complete example of generating synthetic points, building a distance matrix, and calculating delta hyperbolicity:

```python
import numpy as np
from hypdelta import generate_synthetic_points, build_dist_matrix, hypdelta

# Generate synthetic points
dimensions = 100
num_points = 100
points = generate_synthetic_points(dimensions, num_points)

# Build distance matrix
dist_matrix = build_dist_matrix(points)

# Calculate delta hyperbolicity on CPU
