[![PyPI version](https://badge.fury.io/py/poisson-numcodecs.svg)](https://badge.fury.io/py/poisson-numcodecs) ![tests](https://github.com/datajoint/poisson-numcodecs/actions/workflows/tests.yaml/badge.svg)

# Poisson - numcodecs implementation

This codec is designed for compressing movies with Poisson noise, which are produced by photon-limited modalities such multiphoton microscopy, radiography, and astronomy.

The codec assumes that the video is linearly encoded with a potential offset (`dark_signal`) and that the `photon_sensitivity` (the average increase in intensity per photon) is known or can be accurately estimated from the data.

The codec re-quantizes the grayscale efficiently with a square-root-like transformation to equalize the noise variance across the grayscale levels: the [Anscombe Transform](https://en.wikipedia.org/wiki/Anscombe_transform).
This results in a smaller number of unique grayscale levels and improvements in the compressibility of the data with a tunable trade-off (`beta`) for signal accuracy.

To use the codec, one must supply two pieces of information: `dark_signal` (the input value corresponding to the absence of light) and `photon_sensitivity` (levels/photon). We provide two alternative routines to extract those numbers directly from signal statistics. Alternatively, they can be directly measured at the moment of data acquisition. Those calibration routines are provided in the [src/poisson_numcodecs/calibrate.py](src/poisson_numcodecs/calibrate.py) file. 

The codec is used in Zarr as a filter prior to compression.

[Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## Installation

Install via `pip`:

```
pip install poisson-numcodecs
```

### Developer installation

```
conda create -n poisson_numcodecs python=3.xx
conda activate poisson_numcodecs
git clone https://github.com/AllenNeuralDynamics/poisson-numcodecs.git
cd poisson-numcodecs
pip install -r requirements.txt
pip install -e .
```

Make sure everything works:

```
pip install pytest
pytest tests/
```

## Usage

A complete example with sequential calibration and look-up compression is provided in [examples/raster_calibration_and_compression.ipynb](examples/raster_calibration_and_compression.ipynb)
A complete example with raster calibration and compression is provided in [examples/sequential_calibration_and_lookup_compression.ipynb](examples/sequential_calibration_and_lookup_compression.ipynb)