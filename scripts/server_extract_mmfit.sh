#!/usr/bin/env bash
# ==============================================================================
# Turnkey Server Extraction Script for MM-Fit External Benchmark (Protocol A)
# Runs MediaPipe Pose Heavy (model_complexity=2) across 5 Unseen-Test Workouts:
#   w00, w05, w12, w13, w20 (~10.7 GB RGB input -> ~15 MB compressed CSV output)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "MM-FIT MEDIAPIPE POSE HEAVY EXTRACTION (SERVER PIPELINE)"
echo "Project Root: ${PROJECT_ROOT}"
echo "========================================================================"

# 1. Environment Check
echo "[1/4] Checking python environment and dependencies..."
python3 -c "import mediapipe; import cv2; import pandas; import numpy; print('Dependencies verified.')" || {
    echo "Installing missing dependencies..."
    pip install mediapipe opencv-python-headless pandas numpy
}

# 2. Run High-Throughput Extractor
# Downloads 5 videos from Zenodo and extracts landmarks in a single continuous pass
WORKERS=${1:-8}
echo "[2/4] Executing MediaPipe extraction with ${WORKERS} worker processes..."
python3 scripts/extract_external_mediapipe.py \
    --workouts w00 w05 w12 w13 w20 \
    --raw-dir data_external/mmfit/raw/rgb \
    --out-dir data_external/mmfit/landmarks \
    --complexity 2 \
    --workers "${WORKERS}"

# 3. Run Quality Audit and Metadata Generation
echo "[3/4] Running landmark audit and building master segment metadata..."
python3 scripts/audit_mmfit.py \
    --mmfit-dir mm-fit \
    --rgb-dir data_external/mmfit/raw/rgb \
    --landmarks-dir data_external/mmfit/landmarks \
    --out-dir outputs/external/mmfit

# 4. Package Landmarks for High-Speed Local Sync
echo "[4/4] Archiving landmarks for fast transfer to local machine..."
ARCHIVE_PATH="${PROJECT_ROOT}/data_external/mmfit_mediapipe_landmarks.tar.gz"
tar -czvf "${ARCHIVE_PATH}" -C data_external/mmfit landmarks
ARCHIVE_SIZE=$(ls -lh "${ARCHIVE_PATH}" | awk '{print $5}')

echo "========================================================================"
echo "EXTRACTION AND AUDIT COMPLETED SUCCESSFULLY!"
echo "Archive created at: ${ARCHIVE_PATH} (${ARCHIVE_SIZE})"
echo "Transfer command to local machine:"
echo "  scp <user>@<server_ip>:${ARCHIVE_PATH} /Volumes/WorkSpace/Project/Gym_Classification/data_external/"
echo "Then extract locally:"
echo "  tar -xzvf data_external/mmfit_mediapipe_landmarks.tar.gz -C data_external/mmfit/"
echo "========================================================================"
