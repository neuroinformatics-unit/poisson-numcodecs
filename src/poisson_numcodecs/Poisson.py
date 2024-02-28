"""
Numcodecs Codec implementation for Poisson noise calibration
"""
import numpy as np
import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy
from . import estimate

### NUMCODECS Codec ###
class Poisson(Codec):
    """Codec for 3-dimensional data. The codec assumes that input data are of shape:
    (time, x, y).

    Parameters
    ----------
    zero_level : float
        Signal level when no photons are recorded. 
        This should pre-computed or measured directly on the instrument.
    photon_sensitivity : float
        Conversion scalor to convert the measure signal into absolute photon numbers.
        This should pre-computed or measured directly on the instrument.
    """
    codec_id = "poisson"

    def __init__(self, 
                 zero_level, 
                 photon_sensitivity, 
                 encoded_dtype='int8', 
                 decoded_dtype='int16',
                 ): 
        self.zero_level = zero_level
        self.photon_sensitivity = photon_sensitivity
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype

    def encode(self, buf: np.array) -> bytes:
        lookup = estimate.make_anscombe_lookup(self.photon_sensitivity)
        encoded = estimate.lookup(buf, lookup)
        shape = [encoded.ndim] + list(encoded.shape)
        shape = np.array(shape, dtype='uint8')
        return shape.tobytes() + encoded.astype(self.encoded_dtype).tobytes()

    def decode(self, buf: bytes, out=None) -> np.array:
        lookup = estimate.make_anscombe_lookup(self.photon_sensitivity)
        inverse = estimate.make_inverse_lookup(lookup)
        ndims = int(buf[0])
        shape = [int(_) for _ in buf[1:ndims+1]]
        arr = np.frombuffer(buf[ndims+1:], dtype=self.encoded_dtype).reshape(shape)
        decoded = estimate.lookup(arr, inverse)
        return decoded.astype(self.decoded_dtype)


numcodecs.register_codec(Poisson)