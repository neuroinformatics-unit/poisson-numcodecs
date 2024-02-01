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
        self.dtype = np.float32

    def encode(self, buf):
        enc = np.zeros(buf.shape, dtype=self.dtype)
        enc = np.sqrt((buf - self.dark_signal) / self.signal_to_photon_gain)
        return enc

    def decode(self, buf, out=None):
        buf = np.frombuffer(buf, self.dtype)
        dec = buf ** 2 * self.signal_to_photon_gain + self.dark_signal
        out = ndarray_copy(dec, out)
        return out

    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            dark_signal=self.dark_signal,
            signal_to_photon_gain=self.signal_to_photon_gain
        )

numcodecs.register_codec(Poisson)