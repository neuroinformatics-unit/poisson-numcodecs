[![PyPI version](https://badge.fury.io/py/poisson-numcodecs.svg)](https://badge.fury.io/py/poisson-numcodecs) ![tests](https://github.com/datajoint/poisson-numcodecs/actions/workflows/tests.yaml/badge.svg)

# Poisson - numcodecs implementation

This codec is designed for compressing movies with Poisson noise, which are produced by photon-limited modalities such multiphoton microscopy, radiography, and astronomy.

The codec assumes that the video is linearly encoded with a potential offset (`zero_level`) and that the `photon_sensitivity` (the average increase in intensity per photon) is known or can be accurately estimated from the data.

The codec re-quantizes the grayscale efficiently with a square-root-like transformation to equalize the noise variance across the grayscale levels: the [Anscombe Transform](https://en.wikipedia.org/wiki/Anscombe_transform).
This results in a smaller number of unique grayscale levels and significant improvements in the compressibility of the data without sacrificing signal accuracy.

To use the codec, one must supply two pieces of information: `zero_level` (the input value corresponding to the absence of light) and `photon_sensitivity` (levels/photon).

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

An complete example is provided in [examples/workbook.ipynb](examples/workbook.ipynb)
