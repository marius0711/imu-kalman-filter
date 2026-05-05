# IMU Signal Processing & Extended Kalman Filter

Orientation estimation from 6-axis IMU data using an Extended Kalman Filter (EKF).
Validated on the [EuRoC MAV Dataset](https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf) (MH_01_easy sequence).

This project is part of a robotics sensing portfolio — see also
[ros2-imu-fusion](https://github.com/yourusername/ros2-imu-fusion) and
[sensor-calibration](https://github.com/yourusername/sensor-calibration).

---

## What this implements

**Week 1 — Noise characterization & baseline filter**
- Raw IMU data loading and visualization (EuRoC HDF5/CSV format)
- Noise characterization: white noise, bias instability, random walk
- Allan deviation analysis (`allantools`) — quantifies noise floor per axis
- Complementary filter as a sanity-check baseline

**Week 2 — Extended Kalman Filter**
- State vector: quaternion `[q0,q1,q2,q3]` + gyro bias `[bx,by,bz]`
- Process model: quaternion kinematics driven by gyroscope
- Measurement model: gravity reference from accelerometer
- Validation: RMSE < 2° on synthetic circular motion with injected noise
- Visualization with `rerun.io` + matplotlib

---

## Repo structure

```
imu-kalman-filter/
├── data/
│   └── download_euroc.sh     # fetch MH_01_easy (~1.6 GB)
├── src/
│   ├── euroc_loader.py       # IMU + ground-truth CSV reader
│   ├── plot_raw_imu.py       # six-panel raw signal visualization
│   ├── allan_deviation.py    # noise characterization (Week 1)
│   ├── complementary.py      # baseline filter (Week 1)
│   └── ekf.py                # Extended Kalman Filter (Week 2)
├── tests/
│   └── test_euroc_loader.py  # pytest suite (runs without real data)
├── results/                  # generated figures (git-ignored)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/imu-kalman-filter
cd imu-kalman-filter

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download dataset (~1.6 GB, ETH ASL server)
bash data/download_euroc.sh
```

---

## Usage

```bash
# Plot raw IMU signals
python src/plot_raw_imu.py --dataset data/MH_01_easy

# Save figure
python src/plot_raw_imu.py --dataset data/MH_01_easy --save results/raw_imu.png

# Run tests (no dataset required)
pytest tests/ -v
```

---

## Dataset

EuRoC MAV Dataset, MH_01_easy sequence.
Burri et al., *The EuRoC micro aerial vehicle datasets*, IJRR 2016.
IMU: ADIS16448, 200 Hz. Ground truth: Leica Nova MS50 total station.

---

## Background

My bachelor's thesis measured structural acoustics (accelerometer + signal processing pipeline),
and I spent two years doing E/E integration at Magna Steyr — sensor wiring, CAN diagnostics,
test validation. This project applies the same measurement-first thinking to robotics state
estimation: characterize the noise before trusting the filter.
