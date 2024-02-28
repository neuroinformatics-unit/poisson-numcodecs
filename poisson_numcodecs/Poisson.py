"""
Numcodecs Codec implementation for Poisson noise calibration
"""
import numpy as np
import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy, ensure_ndarray

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
                 beta=0.5,
                 ):
        
        self.dark_signal = dark_signal
        self.signal_to_photon_gain = signal_to_photon_gain
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype
        self.background_noise_mean = (
                    -dark_signal / signal_to_photon_gain
                )
        self.beta = beta

    def encode(self, buf):
        enc = np.zeros(buf.shape, dtype=self.encoded_dtype)
        centered = (buf.astype('float') - self.background_noise_mean) / self.signal_to_photon_gain
        enc = 2.0 / self.beta * (np.sqrt(np.maximum(0, centered)))
        enc = enc.astype(self.encoded_dtype)
        return enc

    def decode(self, buf, out=None):
        dec = ensure_ndarray(buf).view(self.encoded_dtype)
        dec = ((dec.astype('float') * self.beta / 2.0 )**2 ) * self.signal_to_photon_gain + self.background_noise_mean
        outarray = np.round(dec)
        outarray = ndarray_copy(outarray, out)
        return outarray.astype(self.decoded_dtype)

numcodecs.register_codec(Poisson)
