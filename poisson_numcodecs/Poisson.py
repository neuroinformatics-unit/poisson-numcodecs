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
    (time, x, y).

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

    def __init__(self, dark_signal, 
                 signal_to_photon_gain, 
                 encoded_dtype='int8', 
                 decoded_dtype='int16',
                 integer_per_photon=4,
                 ):
        
        self.dark_signal = dark_signal
        self.signal_to_photon_gain = signal_to_photon_gain
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype
        self.integer_per_photon = integer_per_photon

    def encode(self, buf):
        print("Buffer: ", buf)
        enc = np.zeros(buf.shape, dtype=self.encoded_dtype)
        centered = (buf.astype('float') - self.dark_signal) / self.signal_to_photon_gain
        enc = self.integer_per_photon * (np.sqrt(np.maximum(0, centered)))
        enc = enc.astype(self.encoded_dtype)
        print("Encoded: ", enc)
        return enc

    def decode(self, buf, out=None):
        print("decoded buffer: ", buf)
        dec = ((buf.astype('float') / self.integer_per_photon)**2) * self.signal_to_photon_gain + self.dark_signal
        outarray = np.round(dec)
        outarray = ndarray_copy(outarray, out)
        print("Decoded: ", outarray.astype(self.decoded_dtype))
        return outarray.astype(self.decoded_dtype)

    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            dark_signal=self.dark_signal,
            signal_to_photon_gain=self.signal_to_photon_gain,
            encoded_dtype=self.encoded_dtype,
            decoded_dtype=self.decoded_dtype,
            integer_per_photon=self.integer_per_photon
        )

numcodecs.register_codec(Poisson)