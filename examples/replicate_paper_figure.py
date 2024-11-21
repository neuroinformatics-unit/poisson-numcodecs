import argparse
from matplotlib import pyplot as plt
import numpy as np
from poisson_numcodecs import Poisson, calibrate
import tiffile as tif
from pathlib import Path
import colorcet as cc


def make_figure(scan: np.ndarray, figure_filename: Path, title: str = None):
    """
    Create a figure with four subplots showing the average, photon transfer curve, 
    coefficient of variation, and quantum flux.

    Method adapted from the original notebook in:
    https://github.com/datajoint/compress-multiphoton/blob/main/notebooks/EvaluatePhotonSensitivity.ipynb

    Parameters
    ----------
    scan : np.ndarray
        Multi-photon movie data.
    figure_filename : Path
        Path to save the output .png file.
    title : str, optional
        Optional title for the figure, by default None
    """
    calibrator = calibrate.SequentialCalibratePhotons(scan)
    [photon_sensitivity, dark_signal] = calibrator.get_photon_sensitivity_parameters()
    print(
        "Quantal size: {sensitivity}\nIntercept: {zero_level}\n".format(
            sensitivity=photon_sensitivity, zero_level=dark_signal
        )
    )

    fig, axx = plt.subplots(2, 2, figsize=(8, 12), tight_layout=True)
    q = photon_sensitivity
    b = dark_signal
    axx = iter(axx.flatten())

    ax = next(axx)
    m = scan.mean(axis=0)
    _ = ax.imshow(m, vmin=0, vmax=np.quantile(m, 0.999), cmap="gray")
    ax.axis(False)
    plt.colorbar(_, ax=ax, ticks=[0.05, 0.5, 0.95]).remove()
    ax.set_title("average")
    ax.text(
        -0.1,
        1.15,
        "A",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ax = next(axx)
    x = np.arange(calibrator.min_intensity, calibrator.max_intensity)
    fit = calibrator.fitted_model.predict(x.reshape(-1, 1))
    ax.scatter(x, np.minimum(fit[-1] * 2, calibrator.fitted_pixels_var), s=2, alpha=0.5)
    ax.plot(x, fit, "r")
    ax.grid(True)
    ax.set_xlabel("intensity")
    ax.set_ylabel("variance")
    ax.set_title("Photon Transfer Curve")
    ax.text(
        -0.1,
        1.15,
        "B",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ax = next(axx)
    v = ((scan[1:, :, :].astype("float64") - scan[:-1, :, :]) ** 2 / 2).mean(axis=0)
    imx = np.stack(((m - b) / q, v / q / q, (m - b) / q), axis=-1)
    _ = ax.imshow(
        np.minimum(
            1, np.sqrt(0.01 + np.maximum(0, imx / np.quantile(imx, 0.9999))) - 0.1
        ),
        cmap="PiYG",
    )
    cbar = plt.colorbar(_, ax=ax, ticks=[0.2, 0.5, 0.8])
    cbar.ax.set_yticklabels(["<< 1", "1", ">> 1"])
    ax.axis(False)
    ax.set_title("coefficient of variation")
    ax.text(
        -0.1,
        1.15,
        "C",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ax = next(axx)
    im = (scan.mean(axis=0) - dark_signal) / photon_sensitivity
    mx = np.quantile(im, 0.999)
    _ = ax.imshow(im, vmin=-mx, vmax=mx, cmap=cc.cm.CET_D13)
    plt.colorbar(_, ax=ax)
    ax.axis(False)
    ax.set_title("Quantum flux\nphotons / pixel / frame")
    ax.text(
        -0.1,
        1.15,
        "D",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plt.suptitle(
        f"{title or figure_filename}\nPhoton sensitivity: {photon_sensitivity:4.1f}"
    )
    fig.savefig(figure_filename, dpi=300)


def main():
    """
    Parse command line arguments and call make_figure.

    Example usage:
    python examples/replicate_paper_figure.py data.tif output_path \
        --title "Title of the figure" --is_negative

    Raises
    ------
    FileNotFoundError
        If the tif file does not exist.
    """

    parser = argparse.ArgumentParser(description="Process and visualize 3-photon data.")
    parser.add_argument("input_tif", type=Path, help="Path to the input .tif file.")
    parser.add_argument(
        "output_path", type=Path, help="Path to save the output .png file."
    )
    parser.add_argument("--title", type=str, help="Optional title for the figure.", default=None)
    parser.add_argument(
        "--is_negative",
        action="store_true",
        help="Flag to indicate if the input data is a.u. and stored as negative.",
    )

    args = parser.parse_args()

    input_path = args.input_tif
    output_path = args.output_path
    title = args.title
    is_negative = args.is_negative

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    data = tif.imread(input_path)

    if is_negative:
        data = data + np.min(data) * -1

    make_figure(data, output_path, title)


if __name__ == "__main__":
    main()
