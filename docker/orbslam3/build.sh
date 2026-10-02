#!/usr/bin/env bash
# Build the ORB-SLAM3 image NavHarness's SLAM stack launches (about 20 minutes, CPU only).
set -euo pipefail
cd "$(dirname "$0")"
docker build -t navharness/orbslam3:latest "$@" .
docker run --rm navharness/orbslam3:latest python3 -c "import orbslam3; print('orbslam3 OK')"
