"""
allan_deviation.py — IMU noise characterization via Allan deviation.

Identifies three noise regimes from the log-log slope:
  slope = -0.5  →  angle random walk (white noise)
  slope =  0.0  →  bias instability
  slope = +0.5  →  rate random walk

Usage:
    python src/allan_deviation.py --dataset data/MH_01_easy
    python src/allan_deviation.py --dataset data/MH_01_easy --save results/allan.png
"""

import argparse
import sys
from pathlib import Path
from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import allantools

sys.path.insert(0, str(Path(__file__).parent))
from euroc_loader import load_imu


AXIS_LABELS = ["X", "Y", "Z"]
GYRO_COLOR  = ["#1d4ed8", "#2563eb", "#60a5fa"]
ACCEL_COLOR = ["#b91c1c", "#dc2626", "#f87171"]


def compute_adev(signal: np.ndarray, rate: float) -> tuple:
    """
    Compute overlapping Allan deviation for a single axis.

    Returns
    -------
    tau : np.ndarray — averaging times [s]
    adev : np.ndarray — Allan deviation values
    """
    result = allantools.oadev(signal, rate=rate, data_type="freq")
    tau  = np.array(result[0])
    adev = np.array(result[1])
    return tau, adev


def estimate_noise_params(tau: np.ndarray, adev: np.ndarray) -> dict:
    """
    Estimate ARW and bias instability from the Allan deviation curve.

    ARW  : Allan deviation at tau=1s  (slope ~ -0.5 region)
    BI   : minimum of the Allan deviation curve
    """
    arw = float(np.interp(1.0, tau, adev))
    bi  = float(np.min(adev))
    tau_bi = float(tau[np.argmin(adev)])
    return {"arw": arw, "bias_instability": bi, "tau_bi": tau_bi}


def plot_allan(imu: dict, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Two-panel Allan deviation plot: gyroscope (left) + accelerometer (right).
    Log-log scale. Reference slope lines for ARW and bias instability.
    """
    rate = 1.0 / np.diff(imu["t"]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Allan Deviation — IMU Noise Characterization", fontsize=13, fontweight="bold")

    sensors = [
        ("Gyroscope",     imu["gyro"],  GYRO_COLOR,  "rad/s"),
        ("Accelerometer", imu["accel"], ACCEL_COLOR, "m/s²"),
    ]

    for ax, (name, data, colors, unit) in zip(axes, sensors):
        params_all = []
        for i, (label, color) in enumerate(zip(AXIS_LABELS, colors)):
            tau, adev = compute_adev(data[:, i], rate)
            ax.loglog(tau, adev, color=color, linewidth=1.5, label=f"Axis {label}")
            params_all.append(estimate_noise_params(tau, adev))

        # Reference slope lines
        tau_ref = np.array([tau[0], tau[-1]])
        arw_mean = np.mean([p["arw"] for p in params_all])
        bi_mean  = np.mean([p["bias_instability"] for p in params_all])
        tau_bi_mean = np.mean([p["tau_bi"] for p in params_all])

        # ARW slope (-0.5): ADEV = ARW / sqrt(tau)
        ax.loglog(tau_ref, arw_mean / np.sqrt(tau_ref),
                  "k--", linewidth=0.8, alpha=0.5, label="ARW slope (−½)")

        # Bias instability marker
        ax.axhline(bi_mean, color="gray", linewidth=0.8, linestyle=":",
                   label=f"BI ≈ {bi_mean:.2e} {unit}")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Averaging time τ [s]")
        ax.set_ylabel(f"Allan Deviation [{unit}]")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

        # Print params to console
        print(f"\n  {name}")
        for i, p in enumerate(params_all):
            print(f"    Axis {AXIS_LABELS[i]}: "
                  f"ARW={p['arw']:.4e} {unit}  "
                  f"BI={p['bias_instability']:.4e} {unit}  "
                  f"(τ_BI={p['tau_bi']:.1f}s)")

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n[INFO] Figure saved to {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Allan deviation for IMU noise characterization.")
    parser.add_argument("--dataset", default="data/MH_01_easy")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    print(f"[INFO] Loading IMU data from {args.dataset} ...")
    imu = load_imu(args.dataset)
    print(f"[INFO] {len(imu['t'])} samples, {1/np.diff(imu['t']).mean():.1f} Hz")
    print("\nAllan Deviation noise parameters:")
    plot_allan(imu, save_path=args.save)


if __name__ == "__main__":
    main()
