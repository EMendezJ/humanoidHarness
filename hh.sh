#!/bin/bash
# Run a humanoidHarness script on the G1's Jetson with its environment:
#   ~/humanoidHarness/hh.sh check_real.py status
#   ~/humanoidHarness/hh.sh agent.py --backend real --arms left
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tv
export PYTHONNOUSERSITE=1 HH_IFACE=${HH_IFACE:-eth0} HH_CAM_HOST=${HH_CAM_HOST:-127.0.0.1} HH_MARKER_SIZE=${HH_MARKER_SIZE:-0.17} HH_SEG_HOST=${HH_SEG_HOST:-127.0.0.1}
# Camera position correction measured with check_real.py arm --aruco (file: "dx,dy,dz" in m).
[ -f camera_offset ] && export FABLE_PICK_CAM_OFFSET=$(cat camera_offset)
# Full 6-DoF calibration from check_real.py calib; when present it replaces camera_offset.
[ -f camera_calib.json ] && export FABLE_PICK_CAM_CALIB=$PWD/camera_calib.json
exec .venv/bin/python "$@"
