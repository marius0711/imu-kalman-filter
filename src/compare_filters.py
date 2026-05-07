"""
Filter comparison: accel-only tilt vs. complementary filter vs. EKF.

Runs all three estimators on the same synthetic pitch-sweep dataset and
produces a four-panel figure:
  - Panel 1: Estimated pitch vs. ground truth for all three filters
  - Panel 2: Angular error over time for each filter
  - Panel 3: EKF gyro bias estimate (x, y, z)
  - Panel 4: Cumulative RMSE convergence

Usage:
  python src/compare_filters.py [--dataset DATA_DIR] [--save OUTPUT.png]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ekf import EKF, quat_angle_error_deg

try:
    from complementary_filter import ComplementaryFilter
    _HAS_CF_MODULE = True
except ImportError:
    _HAS_CF_MODULE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 200.0
DT = 1.0 / FS
G = 9.81

SIGMA_GYRO = 1.7e-4
SIGMA_BIAS = 1.9e-5
SIGMA_ACCEL = 2.0e-3


def generate_pitch_sweep(
    omega_y: float = 0.2,
    duration: float = 20.0,
    seed: int = 42,
) -> dict:
    """Synthetic constant-rate pitch rotation with realistic IMU noise."""
    rng = np.random.default_rng(seed)
    N = int(duration * FS)
    t = np.arange(N) * DT
    theta = omega_y * t

    q_true = np.zeros((N, 4))
    q_true[:, 0] = np.cos(theta / 2.0)
    q_true[:, 2] = np.sin(theta / 2.0)

    gyro_sigma = SIGMA_GYRO * np.sqrt(FS)
    accel_sigma = SIGMA_ACCEL * np.sqrt(FS)

    gyro = np.zeros((N, 3))
    gyro[:, 1] = omega_y
    gyro += rng.normal(0, gyro_sigma, (N, 3))

    accel = np.stack(
        [-G * np.sin(theta), np.zeros(N), G * np.cos(theta)], axis=1
    )
    accel += rng.normal(0, accel_sigma, (N, 3))

    return {
        "timestamps": t,
        "gyro": gyro,
        "accel": accel,
        "q_true": q_true,
        "pitch_true_deg": np.degrees(theta),
    }


def run_accel_only(accel: np.ndarray) -> np.ndarray:
    """
    Tilt-only pitch from accelerometer (no gyro).

    pitch = arctan2(ax, sqrt(ay^2 + az^2))  is not quite right for pure Y rotation.
    For rotation around Y: accel = [g*sin(theta), 0, g*cos(theta)]
    => theta = arctan2(ax, az)
    """
    ax = accel[:, 0]
    az = accel[:, 2]
    return np.degrees(np.arctan2(-ax, az))


def run_complementary_filter(
    gyro: np.ndarray,
    accel: np.ndarray,
    timestamps: np.ndarray,
    alpha: float = 0.98,
) -> np.ndarray:
    """
    Run complementary filter and return pitch angle array.

    Uses the ComplementaryFilter class from complementary_filter.py if available,
    otherwise falls back to a minimal inline implementation.
    """
    N = len(timestamps)
    pitch_deg = np.zeros(N)

    if _HAS_CF_MODULE:
        cf = ComplementaryFilter(alpha=alpha, dt=DT, g=G)
        for i in range(N):
            cf.update(gyro[i], accel[i])
            pitch_deg[i] = cf.pitch_deg
    else:
        # Inline complementary filter: pitch around Y axis
        # accel-derived pitch: arctan2(ax, az)
        accel_pitch = np.degrees(np.arctan2(accel[:, 0], accel[:, 2]))
        pitch = accel_pitch[0]
        for i in range(N):
            dt_i = DT if i == 0 else float(timestamps[i] - timestamps[i - 1])
            gyro_pitch = np.degrees(gyro[i, 1])
            pitch = alpha * (pitch + gyro_pitch * dt_i) + (1.0 - alpha) * accel_pitch[i]
            pitch_deg[i] = pitch

    return pitch_deg


def run_ekf(
    gyro: np.ndarray,
    accel: np.ndarray,
    timestamps: np.ndarray,
) -> tuple:
    """Run EKF and return (pitch_deg, quaternions, biases)."""
    N = len(timestamps)
    pitch_deg = np.zeros(N)
    quats = np.zeros((N, 4))
    biases = np.zeros((N, 3))

    ekf = EKF(
        sigma_gyro=SIGMA_GYRO,
        sigma_bias=SIGMA_BIAS,
        sigma_accel=SIGMA_ACCEL,
        fs=FS,
        g=G,
    )
    quats[0] = ekf.get_quaternion()
    _, pitch_deg[0], _ = ekf.get_euler_deg()
    biases[0] = ekf.get_bias()

    for i in range(1, N):
        dt = float(timestamps[i] - timestamps[i - 1])
        ekf.predict(gyro[i], dt)
        ekf.update(accel[i])
        _, pitch_deg[i], _ = ekf.get_euler_deg()
        quats[i] = ekf.get_quaternion()
        biases[i] = ekf.get_bias()

    return pitch_deg, quats, biases


def cumulative_rmse(errors: np.ndarray) -> np.ndarray:
    """Running RMSE over time."""
    cumsum_sq = np.cumsum(errors ** 2)
    counts = np.arange(1, len(errors) + 1)
    return np.sqrt(cumsum_sq / counts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_figure(data: dict, save_path: str = None) -> None:
    t = data["timestamps"]
    gyro = data["gyro"]
    accel = data["accel"]
    q_true = data["q_true"]
    pitch_true = data["pitch_true_deg"]

    print("Running accel-only estimator...")
    pitch_accel = run_accel_only(accel)

    print("Running complementary filter...")
    pitch_cf = run_complementary_filter(gyro, accel, t)

    print("Running EKF...")
    pitch_ekf, quats_ekf, biases_ekf = run_ekf(gyro, accel, t)

    # Angular errors (pitch only for accel & CF; full quaternion for EKF)
    err_accel = np.abs(pitch_accel - pitch_true)
    err_cf = np.abs(pitch_cf - pitch_true)

    q_ekf_interp = quats_ekf
    err_ekf_quat = quat_angle_error_deg(q_ekf_interp, q_true)

    print(f"\n--- RMSE comparison (skipping first 2 s) ---")
    skip = int(2.0 * FS)
    rmse_accel = float(np.sqrt(np.mean(err_accel[skip:] ** 2)))
    rmse_cf = float(np.sqrt(np.mean(err_cf[skip:] ** 2)))
    rmse_ekf = float(np.sqrt(np.mean(err_ekf_quat[skip:] ** 2)))
    print(f"  Accel-only:           {rmse_accel:.3f} deg")
    print(f"  Complementary filter: {rmse_cf:.3f} deg")
    print(f"  EKF:                  {rmse_ekf:.3f} deg")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    C_TRUE = "#aaaaaa"
    C_ACCEL = "#e07b39"
    C_CF = "#4fc3f7"
    C_EKF = "#a5d6a7"
    ALPHA = 0.85

    # --- Panel 1: Pitch vs ground truth ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#181c25")
    ax1.plot(t, pitch_true, color=C_TRUE, lw=1.5, label="Ground truth", zorder=3)
    ax1.plot(t, pitch_accel, color=C_ACCEL, lw=1.0, alpha=ALPHA, label="Accel-only")
    ax1.plot(t, pitch_cf, color=C_CF, lw=1.0, alpha=ALPHA, label="Comp. filter")
    ax1.plot(t, pitch_ekf, color=C_EKF, lw=1.2, alpha=ALPHA, label="EKF")
    ax1.set_xlabel("Time [s]", color="#aaaaaa", fontsize=9)
    ax1.set_ylabel("Pitch [deg]", color="#aaaaaa", fontsize=9)
    ax1.set_title("Pitch estimation", color="#dddddd", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, framealpha=0.3, loc="upper left")
    ax1.tick_params(colors="#777777", labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")

    # --- Panel 2: Angular error ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#181c25")
    ax2.plot(t, err_accel, color=C_ACCEL, lw=0.8, alpha=0.7, label=f"Accel  RMSE={rmse_accel:.2f}°")
    ax2.plot(t, err_cf, color=C_CF, lw=0.8, alpha=0.7, label=f"CF     RMSE={rmse_cf:.2f}°")
    ax2.plot(t, err_ekf_quat, color=C_EKF, lw=1.0, alpha=0.9, label=f"EKF    RMSE={rmse_ekf:.2f}°")
    ax2.axhline(2.0, color="#ff6b6b", lw=0.8, ls="--", label="2° target")
    ax2.set_xlabel("Time [s]", color="#aaaaaa", fontsize=9)
    ax2.set_ylabel("Angular error [deg]", color="#aaaaaa", fontsize=9)
    ax2.set_title("Error over time", color="#dddddd", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7, framealpha=0.3, loc="upper right")
    ax2.tick_params(colors="#777777", labelsize=8)
    ax2.set_ylim(bottom=0)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    # --- Panel 3: EKF gyro bias estimate ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#181c25")
    labels = ["bx", "by", "bz"]
    colors_b = ["#ef9a9a", "#90caf9", "#ce93d8"]
    for i, (lbl, col) in enumerate(zip(labels, colors_b)):
        ax3.plot(t, biases_ekf[:, i] * 1e3, color=col, lw=0.9, alpha=0.85, label=lbl)
    ax3.set_xlabel("Time [s]", color="#aaaaaa", fontsize=9)
    ax3.set_ylabel("Bias estimate [mrad/s]", color="#aaaaaa", fontsize=9)
    ax3.set_title("EKF bias estimate", color="#dddddd", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7, framealpha=0.3, ncol=3)
    ax3.tick_params(colors="#777777", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333333")

    # --- Panel 4: Cumulative RMSE convergence ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#181c25")
    ax4.plot(t, cumulative_rmse(err_accel), color=C_ACCEL, lw=0.9, label="Accel-only")
    ax4.plot(t, cumulative_rmse(err_cf), color=C_CF, lw=0.9, label="Comp. filter")
    ax4.plot(t, cumulative_rmse(err_ekf_quat), color=C_EKF, lw=1.1, label="EKF")
    ax4.axhline(2.0, color="#ff6b6b", lw=0.8, ls="--", label="2° target")
    ax4.set_xlabel("Time [s]", color="#aaaaaa", fontsize=9)
    ax4.set_ylabel("Cumulative RMSE [deg]", color="#aaaaaa", fontsize=9)
    ax4.set_title("Convergence (cumulative RMSE)", color="#dddddd", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=7, framealpha=0.3)
    ax4.tick_params(colors="#777777", labelsize=8)
    ax4.set_ylim(bottom=0)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#333333")

    fig.suptitle(
        "IMU Filter Comparison — Accel-only vs. Complementary vs. EKF",
        color="#eeeeee",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare IMU orientation filters.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional path to EuRoC-format dataset (uses synthetic data if omitted).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save figure to this path (e.g. results/filter_comparison.png).",
    )
    args = parser.parse_args()

    print("Generating synthetic pitch-sweep dataset (omega_y=0.2 rad/s, 20 s)...")
    data = generate_pitch_sweep(omega_y=0.2, duration=20.0)

    if args.save:
        os.makedirs(os.path.dirname(args.save) if os.path.dirname(args.save) else ".", exist_ok=True)

    build_figure(data, save_path=args.save)
