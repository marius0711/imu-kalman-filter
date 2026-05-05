from typing import Union, Optional
"""
generate_synthetic_imu.py — Generate synthetic IMU data mimicking EuRoC MH_01_easy.

Noise parameters taken from EuRoC paper (Burri et al., IJRR 2016):
  Gyro  noise density : 1.7e-4 rad/s/sqrt(Hz)
  Gyro  bias walk     : 1.9e-5 rad/s^2/sqrt(Hz)
  Accel noise density : 2.0e-3 m/s^2/sqrt(Hz)
  Accel bias walk     : 3.0e-3 m/s^3/sqrt(Hz)

Motion: sinusoidal rotation + slow translation (roughly mimics MH_01 MAV flight).

Usage:
    python src/generate_synthetic_imu.py
    python src/generate_synthetic_imu.py --duration 120 --out data/MH_01_easy
"""

import argparse
from pathlib import Path

import numpy as np


# EuRoC ADIS16448 noise parameters
GYRO_NOISE_DENSITY  = 1.7e-4   # rad/s/sqrt(Hz)
GYRO_BIAS_WALK      = 1.9e-5   # rad/s^2/sqrt(Hz)
ACCEL_NOISE_DENSITY = 2.0e-3   # m/s^2/sqrt(Hz)
ACCEL_BIAS_WALK     = 3.0e-3   # m/s^3/sqrt(Hz)
GRAVITY             = 9.81     # m/s^2
SAMPLE_RATE         = 200.0    # Hz


def generate(duration: float = 60.0, seed: int = 42) -> dict:
    """
    Generate synthetic 6-axis IMU data.

    Parameters
    ----------
    duration : float
        Length of sequence in seconds.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        't'     : (N,) float64 — time in seconds
        'gyro'  : (N,3) float64 — angular velocity [rad/s]
        'accel' : (N,3) float64 — linear acceleration [m/s^2]
        'quat'  : (N,4) float64 — ground-truth quaternion (qw,qx,qy,qz)
        'bias_gyro'  : (N,3) float64 — true gyro bias
        'bias_accel' : (N,3) float64 — true accel bias
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / SAMPLE_RATE
    N = int(duration * SAMPLE_RATE)
    t = np.arange(N) * dt

    # --- True angular velocity (slow sinusoidal motion) ---
    omega_true = np.column_stack([
        0.3 * np.sin(2 * np.pi * 0.05 * t),   # roll rate
        0.2 * np.sin(2 * np.pi * 0.03 * t + 1.0),  # pitch rate
        0.1 * np.sin(2 * np.pi * 0.02 * t + 0.5),  # yaw rate
    ])

    # --- Integrate to get ground-truth quaternion ---
    quat = np.zeros((N, 4))
    quat[0] = [1.0, 0.0, 0.0, 0.0]  # identity
    for i in range(1, N):
        q = quat[i - 1]
        w = omega_true[i - 1]
        # Quaternion derivative: dq/dt = 0.5 * q ⊗ [0, w]
        wx, wy, wz = w
        omega_mat = 0.5 * np.array([
            [  0, -wx, -wy, -wz],
            [ wx,   0,  wz, -wy],
            [ wy, -wz,   0,  wx],
            [ wz,  wy, -wx,   0],
        ])
        q_new = q + dt * omega_mat @ q
        quat[i] = q_new / np.linalg.norm(q_new)

    # --- True acceleration in body frame (gravity + slow translation) ---
    # Rotate gravity vector into body frame using quaternion
    accel_true = np.zeros((N, 3))
    for i in range(N):
        qw, qx, qy, qz = quat[i]
        # Rotation matrix R (world → body)
        R = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [  2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
            [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
        ])
        g_world = np.array([0.0, 0.0, GRAVITY])
        accel_true[i] = R @ g_world  # gravity in body frame

    # --- Bias random walk ---
    gyro_bias = np.cumsum(
        rng.normal(0, GYRO_BIAS_WALK * np.sqrt(dt), (N, 3)), axis=0
    )
    accel_bias = np.cumsum(
        rng.normal(0, ACCEL_BIAS_WALK * np.sqrt(dt), (N, 3)), axis=0
    )
    # Start with small initial bias
    gyro_bias  += rng.normal(0, 0.01, 3)
    accel_bias += rng.normal(0, 0.05, 3)

    # --- White noise ---
    gyro_noise  = rng.normal(0, GYRO_NOISE_DENSITY  * np.sqrt(SAMPLE_RATE), (N, 3))
    accel_noise = rng.normal(0, ACCEL_NOISE_DENSITY * np.sqrt(SAMPLE_RATE), (N, 3))

    # --- Measured signals ---
    gyro  = omega_true  + gyro_bias  + gyro_noise
    accel = accel_true  + accel_bias + accel_noise

    return {
        "t": t,
        "gyro": gyro,
        "accel": accel,
        "quat": quat,
        "bias_gyro": gyro_bias,
        "bias_accel": accel_bias,
    }


def save_euroc_format(data: dict, out_dir: Union[str, Path]) -> None:
    """
    Write CSV files in EuRoC format so euroc_loader.py works unchanged.
    """
    from typing import Union, Optional
    root = Path(out_dir)
    imu_dir = root / "mav0" / "imu0"
    gt_dir  = root / "mav0" / "state_groundtruth_estimate0"
    imu_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    t_ns = (data["t"] * 1e9).astype(np.int64)
    N = len(t_ns)

    # IMU CSV
    imu_path = imu_dir / "data.csv"
    imu_data = np.column_stack([t_ns, data["gyro"], data["accel"]])
    header = "#timestamp [ns],wx [rad s^-1],wy [rad s^-1],wz [rad s^-1]," \
             "ax [m s^-2],ay [m s^-2],az [m s^-2]"
    np.savetxt(imu_path, imu_data, delimiter=",", header=header,
               comments="", fmt=["%d"] + ["%.9f"] * 6)

    # Ground truth CSV (position = zeros, velocity = zeros, no bias cols needed)
    gt_path = gt_dir / "data.csv"
    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    bias_zeros = np.zeros((N, 6))
    gt_data = np.column_stack([t_ns, pos, data["quat"], vel, bias_zeros])
    gt_header = ("#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
                 "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],"
                 "v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],"
                 "b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],"
                 "b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]")
    np.savetxt(gt_path, gt_data, delimiter=",", header=gt_header,
               comments="", fmt=["%d"] + ["%.9f"] * 16)

    print(f"[INFO] Saved IMU data      → {imu_path}")
    print(f"[INFO] Saved ground truth  → {gt_path}")
    print(f"[INFO] Duration: {data['t'][-1]:.1f} s  |  Samples: {N}  |  Rate: 200 Hz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EuRoC-format IMU data.")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Sequence length in seconds (default: 60)")
    parser.add_argument("--out", default="data/MH_01_easy",
                        help="Output directory (default: data/MH_01_easy)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[INFO] Generating {args.duration:.0f}s of synthetic IMU data ...")
    data = generate(duration=args.duration, seed=args.seed)
    save_euroc_format(data, args.out)
    print("[DONE]")


if __name__ == "__main__":
    main()
