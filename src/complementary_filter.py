"""
complementary_filter.py — Complementary filter for orientation estimation.

Fuses gyroscope (high-pass) and accelerometer (low-pass) to estimate
roll and pitch angles. No quaternions — plain Euler angles for clarity.
This is the baseline we compare the EKF against.

Theory:
    angle = alpha * (angle + gyro * dt) + (1 - alpha) * accel_angle
    alpha ~ 0.98: trust gyro 98% short-term, correct with accel 2% each step

Limitations (by design — motivates EKF):
    - Yaw unobservable (accelerometer gives no yaw reference)
    - Fixed alpha: no uncertainty quantification
    - Accel reference breaks under external accelerations (non-gravity forces)
    - No bias estimation

Usage:
    python src/complementary_filter.py --dataset data/MH_01_easy
    python src/complementary_filter.py --dataset data/MH_01_easy --alpha 0.95 --save results/complementary.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from euroc_loader import load_imu


def accel_to_angles(accel: np.ndarray) -> np.ndarray:
    """
    Compute roll and pitch from accelerometer using gravity reference.

    Assumes sensor is approximately stationary (accel ≈ gravity vector).
    Returns angles in degrees: shape (N, 2) — [roll, pitch]

    Physics:
        roll  = atan2(ay, az)   — rotation around X axis
        pitch = atan2(-ax, sqrt(ay² + az²))  — rotation around Y axis
    """
    ax, ay, az = accel[:, 0], accel[:, 1], accel[:, 2]
    roll  = np.degrees(np.arctan2(ay, az))
    pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
    return np.column_stack([roll, pitch])


def run(imu: dict, alpha: float = 0.98) -> dict:
    """
    Run complementary filter on IMU data.

    Parameters
    ----------
    imu   : dict from euroc_loader.load_imu()
    alpha : float — high-pass weight for gyroscope (default 0.98)

    Returns
    -------
    dict with keys:
        't'              : (N,) time vector [s]
        'roll_cf'        : (N,) estimated roll  [deg]
        'pitch_cf'       : (N,) estimated pitch [deg]
        'roll_accel'     : (N,) accel-only roll  [deg]  (for comparison)
        'pitch_accel'    : (N,) accel-only pitch [deg]
        'roll_gyro'      : (N,) gyro-only roll   [deg]  (drifts)
        'pitch_gyro'     : (N,) gyro-only pitch  [deg]  (drifts)
    """
    t     = imu["t"]
    gyro  = imu["gyro"]   # (N,3) rad/s
    accel = imu["accel"]  # (N,3) m/s^2
    N     = len(t)

    # Accelerometer-only angles (noisy but drift-free)
    accel_angles = accel_to_angles(accel)

    # Gyro-only integration (clean short-term, drifts long-term)
    gyro_roll  = np.zeros(N)
    gyro_pitch = np.zeros(N)
    for i in range(1, N):
        dt = t[i] - t[i - 1]
        gyro_roll[i]  = gyro_roll[i-1]  + np.degrees(gyro[i-1, 0]) * dt
        gyro_pitch[i] = gyro_pitch[i-1] + np.degrees(gyro[i-1, 1]) * dt

    # Complementary filter
    cf_roll  = np.zeros(N)
    cf_pitch = np.zeros(N)
    # Initialize from accelerometer
    cf_roll[0]  = accel_angles[0, 0]
    cf_pitch[0] = accel_angles[0, 1]

    for i in range(1, N):
        dt = t[i] - t[i - 1]
        # Gyro contribution: integrate angular velocity
        gyro_roll_step  = cf_roll[i-1]  + np.degrees(gyro[i-1, 0]) * dt
        gyro_pitch_step = cf_pitch[i-1] + np.degrees(gyro[i-1, 1]) * dt
        # Fuse: trust gyro for fast changes, correct drift with accel
        cf_roll[i]  = alpha * gyro_roll_step  + (1 - alpha) * accel_angles[i, 0]
        cf_pitch[i] = alpha * gyro_pitch_step + (1 - alpha) * accel_angles[i, 1]

    return {
        "t":           t,
        "roll_cf":     cf_roll,
        "pitch_cf":    cf_pitch,
        "roll_accel":  accel_angles[:, 0],
        "pitch_accel": accel_angles[:, 1],
        "roll_gyro":   gyro_roll,
        "pitch_gyro":  gyro_pitch,
    }


def plot(result: dict, alpha: float, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Two-panel plot: Roll (top) + Pitch (bottom).
    Three curves each: accel-only, gyro-only, complementary filter.
    """
    t = result["t"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(
        f"Complementary Filter  (α = {alpha})  —  Roll & Pitch Estimation",
        fontsize=13, fontweight="bold"
    )

    panels = [
        ("Roll",  "roll_accel",  "roll_gyro",  "roll_cf"),
        ("Pitch", "pitch_accel", "pitch_gyro", "pitch_cf"),
    ]

    for ax, (name, k_accel, k_gyro, k_cf) in zip(axes, panels):
        ax.plot(t, result[k_accel], color="#94a3b8", linewidth=0.6,
                alpha=0.8, label="Accel only (drift-free, noisy)")
        ax.plot(t, result[k_gyro],  color="#f87171", linewidth=0.8,
                alpha=0.7, label="Gyro only (clean, drifts)")
        ax.plot(t, result[k_cf],    color="#2563eb", linewidth=1.4,
                label=f"Complementary (α={alpha})")
        ax.set_ylabel(f"{name} [deg]")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved to {out}")
    else:
        plt.show()


def print_stats(result: dict) -> None:
    """Print drift comparison between gyro-only and complementary filter."""
    duration = result["t"][-1]
    print(f"\n{'='*55}")
    print(f"  Duration: {duration:.1f} s")
    for name, k_gyro, k_cf in [
        ("Roll",  "roll_gyro",  "roll_cf"),
        ("Pitch", "pitch_gyro", "pitch_cf"),
    ]:
        gyro_drift = abs(result[k_gyro][-1] - result[k_gyro][0])
        cf_drift   = abs(result[k_cf][-1]   - result[k_cf][0])
        print(f"  {name}: gyro drift = {gyro_drift:.2f}°  |  CF drift = {cf_drift:.2f}°")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complementary filter for IMU orientation.")
    parser.add_argument("--dataset", default="data/MH_01_easy")
    parser.add_argument("--alpha",   type=float, default=0.98,
                        help="High-pass weight for gyroscope (default: 0.98)")
    parser.add_argument("--save",    default=None)
    args = parser.parse_args()

    print(f"[INFO] Loading IMU data from {args.dataset} ...")
    imu = load_imu(args.dataset)

    result = run(imu, alpha=args.alpha)
    print_stats(result)
    plot(result, alpha=args.alpha, save_path=args.save)


if __name__ == "__main__":
    main()
