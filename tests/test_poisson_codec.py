from poisson_numcodecs import Poisson
import numpy as np
import pytest

DEBUG = False

def make_poisson_ramp_signals(shape=(10, 1, 1), min_rate=1, max_rate=5, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    x, y, times = shape
    output_array = np.zeros(shape, dtype=dtype)
    for x_ind in range(x):
        for y_ind in range(y):
            output_array[x_ind, y_ind, :] = np.random.poisson(
                np.arange(min_rate, max_rate, (max_rate-min_rate)/times))
    return output_array

@pytest.fixture
def test_data(dtype="int16"):
    test2d = make_poisson_ramp_signals(shape=(50, 1, 1), min_rate=1, max_rate=5, dtype=dtype)
    test2d_long = make_poisson_ramp_signals(shape=(1, 50, 1), min_rate=1, max_rate=5, dtype=dtype)
    return [test2d, test2d_long]

def test_poisson_encode_decode(test_data):
    poisson_codec = Poisson(
        dark_signal=0,
        photon_sensitivity=1.0,
        encoded_dtype='uint8', 
        decoded_dtype='int16'
    )
    
    for example_data in test_data:
        encoded = poisson_codec.encode(example_data)
        decoded = poisson_codec.decode(encoded)
        np.testing.assert_allclose(decoded, example_data, atol=1e-3)

def test_poisson_encode_decode_non_lookup(test_data):
    poisson_codec = Poisson(
        dark_signal=0,
        photon_sensitivity=1.0,
        encoded_dtype='uint8', 
        decoded_dtype='int16',
        use_lookup=False
    )
    
    for example_data in test_data:
        encoded = poisson_codec.encode(example_data)
        decoded = poisson_codec.decode(encoded)
        np.testing.assert_allclose(decoded, example_data, atol=1e-3)


if __name__ == '__main__':
    list_data = test_data("int16")
    test_poisson_encode_decode(list_data)
    test_poisson_encode_decode_non_lookup(list_data)
