[![PyPI version](https://badge.fury.io/py/poisson-numcodecs.svg)](https://badge.fury.io/py/poisson-numcodecs) ![tests](https://github.com/AllenNeuralDynamics/poisson-numcodecs/actions/workflows/python-package.yml/badge.svg)

# Poisson - numcodecs implementation

This codec enables one to equalize noise across imaging levels for shot-noise 
limited imaging. This allows for some lossy compression based on fundamental 
statistics coming from each pixel. Precision can be tuned to adjust the compression 
trade-off. 

[Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## Installation

Install via `pip`:

```
pip install poisson-numcodecs
```

Or from sources:

```
git clone https://github.com/AllenNeuralDynamics/poisson-numcodecs.git
cd poisson-numcodecs
pip install .
```

## Usage

This is a simple example on how to use the `Poisson` codec with `zarr`:

```
from poisson_numcodecs import Poisson

data = ... # any 2D dumpy array
# here we assume that the data has a shape of (num_frames, x, y)

# dark_signal and signal_to_photon_gain must be calculated from the data or 
# measured directly on the instrument.
# dark_signal is the signal level when no photons are recorded. 
# signal_to_photon_gain is the conversion scalor to convert the measure signal 
# into absolute photon numbers. 

# instantiate Poisson object
poisson_filter = Poisson(dark_signal, signal_to_photon_gain, encoded_dtype, decoded_dtype, integer_per_photon)

# using default Zarr compressor
photon_data = zarr.array(data, filters=[poisson_filter])

data_read = photon_data[:]
```
Available `**kwargs` can be browsed with: `Poisson?`

**NOTE:** 
In order to reload in zarr an array saved with the `Poisson`, you just need to have the `poisson_numcodecs` package
installed.