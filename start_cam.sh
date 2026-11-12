#!/bin/bash
# Start (or restart) the humanoidHarness camera server on this G1 Jetson, in the background.
cd ~/humanoidHarness && mkdir -p runs
pkill -f "python cam_server.py --source" 2>/dev/null
sleep 0.5
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tv
export PYTHONNOUSERSITE=1
SCALE=$(cat camera_fx_scale 2>/dev/null || echo 1.0)    # focal-scale correction, see check_real.py fxcal
setsid nohup .venv/bin/python cam_server.py --source rpc --fx-scale "$SCALE" > runs/cam_server.log 2>&1 < /dev/null &
