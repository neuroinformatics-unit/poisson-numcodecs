"""
Numcodecs Codec implementation for Delta codec in 2-dimensions
"""
import numpy as np

import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy


### NUMCODECS Codec ###
class Delta2D(Codec):
    """Codec for 2-dimensional Delta. The codec assumes that input data are of shape:
    (num_samples, num_channels).

    Parameters
    ----------
    dtype : numpy.dtype
        The output dtype
    num_channels : int
        Number of channels (second dimension of the 2D input data). 
        If the signal is chunked in the second dimension, number_of channels need to correspond to 
        the channel block size (e.g., chunks=(None, 10) --> `num_channels=10`). 
    axis : int, optional
        The axis to perform the delta operation.
        If axis=0, it applies a Delta filter in the samples domain, by channel.
        Instead, axis=1 applies a delta filter in the channel domain, by default 0.
    """
    codec_id = "delta2D"

    def __init__(self, dtype, num_channels, axis=1):
        self.dtype = dtype
        self.num_channels = num_channels
        self.axis = axis

    def encode(self, buf):
        assert buf.ndim == 2, "Input data must be 2D"
        assert buf.shape[1] == self.num_channels

        enc = np.zeros(buf.shape, dtype=self.dtype)

        if self.axis == 0:
            enc[0, :] = buf[0, :]
            enc[1:] = np.diff(buf, axis=self.axis)
        elif self.axis == 1:
            enc[:, 0] = buf[:, 0]
            enc[:, 1:] = np.diff(buf, axis=self.axis)
        return enc

    def decode(self, buf, out=None):
        buf = np.frombuffer(buf, self.dtype)
        enc = buf.reshape(-1, self.num_channels)
        
        dec = np.empty_like(enc, dtype=self.dtype)
        np.cumsum(enc, out=dec, axis=self.axis)
        out = ndarray_copy(dec, out)
        
        return out

    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            dtype=np.dtype(self.dtype).str if self.dtype is not None else None,
            num_channels=self.num_channels,
            axis=self.axis
        )


numcodecs.register_codec(Delta2D)