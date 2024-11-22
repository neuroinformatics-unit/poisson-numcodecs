import argparse
import os
import numpy as np
from poisson_numcodecs import Poisson, calibrate
import matplotlib.pyplot as plt
import colorcet as cc
from pathlib import Path
from scipy import ndimage
from matplotlib.colors import hsv_to_rgb
import tifffile as tif


def generate_panels(scan: np.ndarray, output_dir: Path, file_name: str):
    """Generate all figure panels (A–F) and save them."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_filename = output_dir / file_name

    # Compute sensitivity stats
    calibrator = calibrate.SequentialCalibratePhotons(scan)
    [photon_sensitivity, dark_signal] = calibrator.get_photon_sensitivity_parameters()
    print(
        f"{figure_filename}\nQuantal size: {photon_sensitivity:5.1f}\nIntercept: {dark_signal:5.1f}\n"
    )

    fig = plt.figure(figsize=(8, 6))  # Adjust overall figure size
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)

    # Panel A: Mean Fluorescence
    ax = fig.add_subplot(gs[0, 0])
    mean_fluorescence = scan.mean(axis=0)
    ax.imshow(
        mean_fluorescence,
        vmin=0,
        vmax=np.quantile(mean_fluorescence, 0.999),
        cmap="gray",
    )
    ax.axis(False)
    ax.set_title("Mean Fluorescence")

    # Panel B: Sensitivity and Variance
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(calibrator.min_intensity, calibrator.max_intensity)
    fit = calibrator.fitted_model.predict(x.reshape(-1, 1))
    ax1.scatter(
        x,
        np.minimum(fit[-1] * 2, calibrator.fitted_pixels_var),
        s=1,
        color="k",
        alpha=0.5,
    )
    ax1.plot(x, fit, color="red", lw=1)
    ax1.set_ylabel("Variance")
    ax1.grid(True)
    ax1.set_title(
        f"Sensitivity={photon_sensitivity:0.1f}; Zero Level={dark_signal:0.0f}"
    )

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(x, calibrator.counts, color="k")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Density")
    ax2.grid(True)

    # Panel C: Coefficient of Variation
    ax = fig.add_subplot(gs[1, 0])
    variance = ((scan[1:, :, :].astype("float64") - scan[:-1, :, :]) ** 2 / 2).mean(
        axis=0
    )
    cv_image = variance / photon_sensitivity**2
    ax.imshow(np.sqrt(cv_image / np.quantile(cv_image, 0.9999)), cmap="PiYG")
    ax.axis(False)
    ax.set_title("Coefficient of Variation")

    # Panel D: Activity Map and Segmentation
    ax = fig.add_subplot(gs[1, 1])
    flux = (scan - dark_signal) / photon_sensitivity
    activity_map = flux.max(axis=0) - flux.mean(axis=0)
    mask = activity_map > activity_map.max() * 0.5
    labels, _ = ndimage.label(mask)
    hsv = np.stack(
        (
            np.ones_like(activity_map) * 0.3,
            (labels > 0) * 0.4,
            activity_map / activity_map.max(),
        ),
        axis=-1,
    )
    ax.imshow(hsv_to_rgb(hsv))
    ax.axis(False)
    ax.set_title("Activity Map with Segmentation")

    # Panel E: Max Flux
    ax = fig.add_subplot(gs[1, 2])
    max_flux = flux.max(axis=0)
    ax.imshow(max_flux, cmap=cc.cm.CET_R4)
    ax.axis(False)
    ax.set_title("Max Flux (photons/pixel/frame)")

    # Panel F: Fluorescence Traces
    traces = np.stack(
        [flux[:, labels == label].sum(axis=1) for label in np.unique(labels)[1:]]
    )
    ax = fig.add_subplot(gs[1, 2])
    time = np.arange(traces.shape[1]) * 0.12
    for i, trace in enumerate(traces, 1):
        ax.plot(time, trace / 10000 + i, "k")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cell Number")
    ax.set_title("Fluorescence Traces")

    # Save the entire figure
    fig.savefig(figure_filename, dpi=300)

    # Return stats
    return {
        "quantal_size": photon_sensitivity,
        "zero_level": dark_signal,
        "compression_ratio": None,  # Placeholder for video compression
    }


# def compress_video(
#     scan: np.ndarray, output_dir: Path, file_name: str, sensitivity, zero_level
# ):
#     """Compress video and return compression ratio."""
#     zero = int(round(zero_level))
#     LUT1, LUT2 = compress.make_luts(
#         zero_level=0, sensitivity=sensitivity, input_max=scan.max() - zero
#     )
#     compressed = compress.lookup(scan - zero, LUT1)
#     gif_path = output_dir / f"{file_name}.gif"
#     compress.save_movie(compressed, str(gif_path), scale=255 // np.max(compressed))
#     compression_ratio = np.prod(scan.shape) * 2 / os.path.getsize(gif_path)
#     print(f"Compression ratio: {compression_ratio:0.2f}")
#     return compression_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-photon figure panels and compress video."
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to input .npz file containing 'scan' data."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save the output figures and stats."
    )
    parser.add_argument(
        "--file_name", type=str, default="output", help="Base name for output files."
    )
    args = parser.parse_args()

    # Load data
    data = tif.imread(args.input_file)

    if np.min(data) < 0:
        data = data + np.min(data) * -1

    # Generate panels
    stats = generate_panels(data, args.output_dir, args.file_name)

    # # Compress video and add stats
    # stats["compression_ratio"] = compress_video(
    #     data,
    #     args.output_dir,
    #     args.file_name,
    #     stats["quantal_size"],
    #     stats["zero_level"],
    # )

    # Print final stats
    print("Final Statistics:")
    print(stats)


if __name__ == "__main__":
    main()
