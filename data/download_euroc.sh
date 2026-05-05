#!/usr/bin/env bash
# Download EuRoC MAV Dataset — MH_01_easy sequence (IMU + ground truth only)
# Source: https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf
# File hosted by ETH ASL: http://robotics.ethz.ch/~asl-datasets/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}"
TARGET="${DATA_DIR}/MH_01_easy.zip"
EXTRACT_DIR="${DATA_DIR}/MH_01_easy"

URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip"

if [ -d "${EXTRACT_DIR}" ]; then
  echo "[INFO] Dataset already extracted at ${EXTRACT_DIR} — skipping download."
  exit 0
fi

echo "[INFO] Downloading EuRoC MH_01_easy (~1.6 GB) ..."
curl -L --progress-bar -o "${TARGET}" "${URL}"

echo "[INFO] Extracting ..."
unzip -q "${TARGET}" -d "${DATA_DIR}"

echo "[INFO] Cleaning up zip ..."
rm "${TARGET}"

echo "[DONE] Dataset available at ${EXTRACT_DIR}"
echo "       IMU data:         ${EXTRACT_DIR}/mav0/imu0/data.csv"
echo "       Ground truth:     ${EXTRACT_DIR}/mav0/state_groundtruth_estimate0/data.csv"
