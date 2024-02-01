"""
Numcodecs Codec implementation for Poisson noise calibration
"""
import numpy as np

import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy


### NUMCODECS Codec ###

class Poisson(Codec):
    """Codec for 3-dimensional Delta. The codec assumes that input data are of shape:
    (num_samples, num_channels).

    Parameters
    ----------
    dark_signal : float
        Signal level when no photons are recorded. 
        This should pre-computed or measured directly on the instrument.
    signal_to_photon_gain : float
        Conversion scalor to convert the measure signal into absolute photon numbers.
        This should pre-computed or measured directly on the instrument.
    """
    codec_id = "poisson"

    def __init__(self, dark_signal, signal_to_photon_gain):
        self.dark_signal = dark_signal
        self.signal_to_photon_gain = signal_to_photon_gain

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


numcodecs.register_codec(Poisson)