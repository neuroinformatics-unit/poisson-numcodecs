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
        self.beta = beta

    def encode(self, buf):
        enc = np.zeros(buf.shape, dtype=self.encoded_dtype)

        # We convert to photons
        centered = (buf.astype('float') - self.dark_signal) / self.signal_to_photon_gain

        # https://en.wikipedia.org/wiki/Anscombe_transform for the forward
        enc = 2.0 (np.sqrt(np.maximum(0, centered + 3/8)))

        # This is the part that is applied to discard noise in compressed form
        enc = enc / self.beta 
        enc = enc.astype(self.encoded_dtype)
        
        return enc

    def decode(self, buf, out=None):
        dec = ensure_ndarray(buf).view(self.encoded_dtype)

        # We first unapply beta
        dec = dec.astype('float') * self.beta
        
        # https://en.wikipedia.org/wiki/Anscombe_transform for the inverse without bias
        dec = dec**2 / 4.0 - 1/8 + 1/4*np.sqrt(1.5)/dec - 11/8/(dec**2)+5/8*np.sqrt(1.5)/(dec**3) 

        # We convert back to arbitrary pixels
        dec = dec * self.signal_to_photon_gain + self.dark_signal

        # We have to go back to integers
        outarray = np.round(dec)
        
        outarray = ndarray_copy(outarray, out)
        return outarray.astype(self.decoded_dtype)

numcodecs.register_codec(Poisson)
