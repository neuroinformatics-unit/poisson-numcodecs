[![PyPI version](https://badge.fury.io/py/flac-numcodecs.svg)](https://badge.fury.io/py/flac-numcodecs) ![tests](https://github.com/AllenNeuralDynamics/flac-numcodecs/actions/workflows/python-package.yml/badge.svg)

# Delta2D - numcodecs implementation

[Numcodecs](https://numcodecs.readthedocs.io/en/latest/index.html) implementation of the [Delta] filter applied to 
2D input data.

This implementation enables one to apply delta filters on specific dimentions as a filter in 
[Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## Installation

Install via `pip`:

```
pip install delta2d-numcodecs
```

Or from sources:

```
git clone https://github.com/AllenNeuralDynamics/delta2d-numcodecs.git
cd flac-numcodecs
pip install .
```

## Usage

This is a simple example on how to use the `Delta2d` codec with `zarr`:

```
from delta2d_numcodecs import Delta2D

data = ... # any 2D dumpy array
# here we assume that the data has a shape of (num_samples, num_channels)

# instantiate Delta2D in time dimension
delta_time = Delta2D(dtype=data.dtype, axis=0)

# instantiate Delta2D in space dimension
delta_space = Delta2D(dtype=data.dtype, axis=1)

# using default Zarr compressor
z_time = zarr.array(data, filters=[delta_time])
z_space = zarr.array(data, filters=[delta_space])
# apply in both time and space, sequentally
z_time_space = zarr.array(data, filters=[delta_time, delta_space])

data_read = z[:]
```
Available `**kwargs` can be browsed with: `Delta2D?`

**NOTE:** 
In order to reload in zarr an array saved with the `Delta2D`, you just need to have the `delta2d_numcodecs` package
installed.