"""
plot_raw_imu.py — Visualize raw accelerometer and gyroscope data from EuRoC.

Usage:
    python src/plot_raw_imu.py --dataset data/MH_01_easy
    python src/plot_raw_imu.py --dataset data/MH_01_easy --save results/raw_imu.png
"""

import argparse
from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np

from euroc_loader import load_imu


def plot_imu(imu: dict, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Six-panel figure: gyroscope (top row) + accelerometer (bottom row).
    Each column = one axis (X, Y, Z).
    """
    t = imu["t"]
    gyro = imu["gyro"]
    accel = imu["accel"]

    axis_labels = ["X", "Y", "Z"]
    gyro_color = "#2563EB"    # blue
    accel_color = "#DC2626"   # red

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    fig.suptitle("EuRoC MH_01_easy — Raw IMU Data", fontsize=13, fontweight="bold")

    for i, label in enumerate(axis_labels):
        # Gyroscope
        ax_g = axes[0, i]
        ax_g.plot(t, gyro[:, i], color=gyro_color, linewidth=0.5)
        ax_g.set_title(f"Gyroscope {label}", fontsize=10)
        ax_g.set_ylabel("Angular velocity [rad/s]" if i == 0 else "")
        ax_g.grid(True, alpha=0.3)

        # Accelerometer
        ax_a = axes[1, i]
        ax_a.plot(t, accel[:, i], color=accel_color, linewidth=0.5)
        ax_a.set_title(f"Accelerometer {label}", fontsize=10)
        ax_a.set_ylabel("Acceleration [m/s²]" if i == 0 else "")
        ax_a.set_xlabel("Time [s]")
        ax_a.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved to {out}")
    else:
        plt.show()


def print_stats(imu: dict) -> None:
    """Print basic statistics for a quick sanity check."""
    t = imu["t"]
    dt = np.diff(t)
    print(f"\n{'='*50}")
    print(f"  Duration     : {t[-1]:.1f} s")
    print(f"  Samples      : {len(t)}")
    print(f"  Mean dt      : {dt.mean()*1e3:.3f} ms  ({1/dt.mean():.1f} Hz)")
    print(f"  Max dt jitter: {(dt.max()-dt.mean())*1e3:.3f} ms")
    print()
    for name, data in [("Gyro  [rad/s]", imu["gyro"]), ("Accel [m/s²]", imu["accel"])]:
        print(f"  {name}")
        for i, axis in enumerate("XYZ"):
            print(
                f"    {axis}: mean={data[:,i].mean():+.4f}  "
                f"std={data[:,i].std():.4f}  "
                f"min={data[:,i].min():+.4f}  "
                f"max={data[:,i].max():+.4f}"
            )
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw EuRoC IMU data.")
    parser.add_argument(
        "--dataset",
        default="data/MH_01_easy",
        help="Path to EuRoC sequence root (default: data/MH_01_easy)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save figure to this path instead of showing it interactively.",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading IMU data from {args.dataset} ...")
    imu = load_imu(args.dataset)

    print_stats(imu)
    plot_imu(imu, save_path=args.save)


if __name__ == "__main__":
    main()
