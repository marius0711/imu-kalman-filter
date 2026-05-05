"""
visualize_rerun.py — Log IMU orientation to rerun.io for interactive 3D viewing.

Logs:
  - Raw accelerometer + gyroscope time series
  - Complementary filter roll/pitch as scalar timeseries
  - Orientation as a 3D rotation (rerun Transform3D)

Usage:
    python src/visualize_rerun.py --dataset data/MH_01_easy
    python src/visualize_rerun.py --dataset data/MH_01_easy --save results/imu_session.rrd
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from euroc_loader import load_imu
from complementary_filter import run as cf_run


def visualize(imu: dict, cf: dict, save_path=None) -> None:
    try:
        import rerun as rr
    except ImportError:
        print("[ERROR] rerun-sdk not installed. Run: pip install rerun-sdk")
        sys.exit(1)

    rr.init("imu-kalman-filter", spawn=(save_path is None))

    if save_path:
        rr.save(str(save_path))

    t = imu["t"]

    # Log raw IMU signals
    for i, axis in enumerate(["X", "Y", "Z"]):
        rr.log_time_series(
            f"imu/gyro/{axis}",
            times=t,
            values=imu["gyro"][:, i],
        )
        rr.log_time_series(
            f"imu/accel/{axis}",
            times=t,
            values=imu["accel"][:, i],
        )

    # Log complementary filter output
    rr.log_time_series("orientation/roll_cf",    times=t, values=cf["roll_cf"])
    rr.log_time_series("orientation/pitch_cf",   times=t, values=cf["pitch_cf"])
    rr.log_time_series("orientation/roll_accel", times=t, values=cf["roll_accel"])

    # Log 3D orientation (subsample to 50 Hz for performance)
    step = 4
    for i in range(0, len(t), step):
        rr.set_time_seconds("stable_time", t[i])
        roll_r  = np.radians(cf["roll_cf"][i])
        pitch_r = np.radians(cf["pitch_cf"][i])
        # Rotation matrix from roll + pitch (yaw = 0)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll_r), -np.sin(roll_r)],
            [0, np.sin(roll_r),  np.cos(roll_r)],
        ])
        Ry = np.array([
            [ np.cos(pitch_r), 0, np.sin(pitch_r)],
            [0, 1, 0],
            [-np.sin(pitch_r), 0, np.cos(pitch_r)],
        ])
        R = Ry @ Rx
        rr.log(
            "sensor/orientation",
            rr.Transform3D(mat3x3=R),
        )

    if save_path:
        print(f"[INFO] Session saved to {save_path}")
        print(f"       Open with: rerun {save_path}")
    else:
        print("[INFO] rerun viewer launched.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize IMU data with rerun.io.")
    parser.add_argument("--dataset", default="data/MH_01_easy")
    parser.add_argument("--alpha",   type=float, default=0.98)
    parser.add_argument("--save",    default=None,
                        help="Save .rrd session file instead of launching viewer")
    args = parser.parse_args()

    print(f"[INFO] Loading IMU data from {args.dataset} ...")
    imu = load_imu(args.dataset)
    cf  = cf_run(imu, alpha=args.alpha)

    visualize(imu, cf, save_path=args.save)


if __name__ == "__main__":
    main()
