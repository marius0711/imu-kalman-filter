"""
euroc_loader.py — Load EuRoC MAV dataset (IMU + ground truth).

EuRoC CSV format:
  imu0/data.csv       : timestamp[ns], wx, wy, wz [rad/s], ax, ay, az [m/s^2]
  state_gt/data.csv   : timestamp[ns], px, py, pz, qw, qx, qy, qz, vx, vy, vz,
                        bwx, bwy, bwz, bax, bay, baz
"""

from pathlib import Path
from typing import Union
import numpy as np


def load_imu(dataset_root: Union[str, Path]) -> dict:
    """
    Load raw IMU data from EuRoC MH_01_easy (or any sequence).

    Returns
    -------
    dict with keys:
        't'     : (N,) float64 — time in seconds, zeroed at first sample
        'gyro'  : (N,3) float64 — angular velocity [rad/s]  (wx, wy, wz)
        'accel' : (N,3) float64 — linear acceleration [m/s^2] (ax, ay, az)
    """
    path = Path(dataset_root) / "mav0" / "imu0" / "data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"IMU data not found at {path}.\n"
            "Run  data/download_euroc.sh  first."
        )

    raw = np.loadtxt(path, delimiter=",", skiprows=1)

    t_ns = raw[:, 0]
    t = (t_ns - t_ns[0]) * 1e-9  # nanoseconds → seconds, zeroed

    gyro = raw[:, 1:4]   # wx, wy, wz
    accel = raw[:, 4:7]  # ax, ay, az

    return {"t": t, "gyro": gyro, "accel": accel}


def load_ground_truth(dataset_root: Union[str, Path]) -> dict:
    """
    Load ground-truth state estimates from EuRoC.

    Returns
    -------
    dict with keys:
        't'    : (M,) float64 — time in seconds, zeroed at first sample
        'pos'  : (M,3) float64 — position [m]
        'quat' : (M,4) float64 — orientation quaternion (qw, qx, qy, qz)
        'vel'  : (M,3) float64 — velocity [m/s]
    """
    path = (
        Path(dataset_root)
        / "mav0"
        / "state_groundtruth_estimate0"
        / "data.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth data not found at {path}.\n"
            "Run  data/download_euroc.sh  first."
        )

    raw = np.loadtxt(path, delimiter=",", skiprows=1)

    t_ns = raw[:, 0]
    t = (t_ns - t_ns[0]) * 1e-9

    pos = raw[:, 1:4]
    quat = raw[:, 4:8]   # qw, qx, qy, qz
    vel = raw[:, 8:11]

    return {"t": t, "pos": pos, "quat": quat, "vel": vel}
