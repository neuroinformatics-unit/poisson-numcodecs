import argparse
from matplotlib import pyplot as plt
import numpy as np
from poisson_numcodecs import Poisson, calibrate
import tiffile as tif
from pathlib import Path
import colorcet as cc


def convert_movie(scan):

    print("Converting movie to photon flux movie...")
    calibrator = calibrate.SequentialCalibratePhotons(scan)

    print("Calibrating photon sensitivity...")
    [photon_sensitivity, dark_signal] = calibrator.get_photon_sensitivity_parameters()
    print(f"Quantal size: {photon_sensitivity}\nIntercept: {dark_signal}\n")

    print("Getting photon flux movie...")
    print(
        "Performing the calculation photon flux movie = (scan  - dark_signal) / photon_sensitivity"
    )
    photon_counts_per_pixel_per_frame = calibrator.get_photon_flux_movie()

    return photon_counts_per_pixel_per_frame


def main():
    """
    This script converts a movie file into photon flux units per pixel per frame.

    It reads an input TIFF movie file, processes it to ensure all pixel values are non-negative,
    calculates the photon flux for each pixel, and saves the resulting movie to an output TIFF file.

    Usage:
        python convert_movie.py input_movie.tif output_movie.tif

    Arguments:
        input: Path to the input movie file (TIFF format).
        output: Path to save the processed photon flux movie (TIFF format).

    Features:
        - Automatically adjusts the input movie to ensure all pixel values are non-negative.
        - Computes the photon flux for each pixel.
        - Saves the processed movie to the specified output path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input movie file")
    parser.add_argument("output", type=str, help="Output movie file")

    args = parser.parse_args()

    scan = tif.imread(args.input)

    if np.min(scan) < 0:
        scan = scan + np.min(scan) * -1

    flux_movie = convert_movie(scan)

    tif.imsave(args.output, flux_movie)

    print(f"Saved photon flux movie to {args.output}")


if __name__ == "__main__":
    main()
