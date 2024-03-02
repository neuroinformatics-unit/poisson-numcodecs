"""
Numcodecs Codec implementation for Poisson noise calibration
"""
import numpy as np
import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray, ndarray_copy

### NUMCODECS Codec ###
class Poisson(Codec):
    """Codec for 3-dimensional Filter. The codec assumes that input data are of shape:
    (time, x, y).

    Parameters
    ----------
    dark_signal : float
        Signal level when no photons are recorded. 
        This should pre-computed or measured directly on the instrument.
    photon_sensitivity : float
        Conversion scalor to convert the measure signal into absolute photon numbers.
        This should pre-computed or measured directly on the instrument.
    encoded_dtype : str
        The dtype of the encoded data.
    decoded_dtype : str
        The dtype of the decoded data.
    beta : float
        The grayscale quantization step expressed in units of noise std dev
    use_lookup : bool
        If True, use lookup table to encode and decode the data. 
        If False, use the forward and inverse functions to encode and decode the data.
    """
    codec_id = "poisson"

    def __init__(self, 
                 dark_signal, 
                 photon_sensitivity, 
                 encoded_dtype='uint8', 
                 decoded_dtype='int16',
                 beta=0.5,
                 use_lookup=True
                 ): 
        
        self.dark_signal = dark_signal
        self.photon_sensitivity = photon_sensitivity
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype
        self.beta = beta
        self.use_lookup = use_lookup

        if self.use_lookup:
            # produce anscombe lookup_tables
            input_max = np.iinfo(self.decoded_dtype).max
            xx = (np.r_[:input_max + 1] - self.dark_signal) / self.photon_sensitivity

            # JEROME: I do not understand why we need to subtract np.sqrt(3/8) here
            # Also, shouldn't te +3/8 be inside the maximum function as xx is a float?
            forward = 2.0 / self.beta * (np.sqrt(np.maximum(0, xx) + 3/8) - np.sqrt(3/8))
            
            # JEROME : I think this might be better syntax?
            forward = np.round(forward).astype(self.encoded_dtype)
            self.forward_table = forward

            _, inverse = np.unique(self.forward_table, return_index=True)
            inverse += (np.r_[:inverse.size] / 
                        inverse.size * (inverse[-1] - inverse[-2])/2).astype(self.decoded_dtype)
            self.inverse_table = inverse

    def _lookup(self, movie, LUT):
        """
        Apply lookup table LUT to input movie
        """
        return LUT[np.maximum(0, np.minimum(movie, LUT.size-1))]

    def encode(self, buf: np.array):            
        encoded = np.zeros(buf.shape, dtype=self.encoded_dtype)

        if self.use_lookup:
            encoded = self._lookup(buf,  self.forward_table)
        else:
            # We convert to photons
            centered = (buf.astype('float') - self.dark_signal) / self.photon_sensitivity

            # https://en.wikipedia.org/wiki/Anscombe_transform for the forward
            encoded = 2.0 * (np.sqrt(np.maximum(0, centered + 3/8)))

            # This is the part that is applied to discard noise in compressed form
            encoded = encoded / self.beta 

            # We have to go to integers in a clean way
            encoded = np.round(encoded)
        
        return encoded.astype(self.encoded_dtype)
    
    def decode(self, buf, out=None):                   
        dec = ensure_ndarray(buf).view(self.encoded_dtype)
 
        if self.use_lookup:
            decoded = self._lookup(dec,  self.inverse_table)
        else:
            # We first unapply beta
            dec = dec.astype('float') * self.beta
            
            # https://en.wikipedia.org/wiki/Anscombe_transform for the inverse without bias
            dec = dec**2 / 4.0 - 1/8

            # We convert back to arbitrary pixels
            dec = dec * self.photon_sensitivity + self.dark_signal

            # We have to go back to integers
            decoded = np.round(dec)
            
            decoded = ndarray_copy(decoded, out)

        return decoded.astype(self.decoded_dtype)
    
    def get_config(self):
        return dict(
            id=self.codec_id,
            dark_signal=self.dark_signal,
            photon_sensitivity=self.photon_sensitivity,
            encoded_dtype=self.encoded_dtype,
            decoded_dtype=self.decoded_dtype,
            beta=self.beta,
            use_lookup=self.use_lookup
        )

numcodecs.register_codec(Poisson)