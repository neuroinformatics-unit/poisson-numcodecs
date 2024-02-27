from pathlib import Path
import numpy as np

test_data_fp = Path(__file__).parent.joinpath("data", "data1.npz")

def test_loaddata():
    movie = np.load(test_data_fp)