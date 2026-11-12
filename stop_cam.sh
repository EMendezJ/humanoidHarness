#!/bin/bash
# Stop the humanoidHarness camera server (only ours; rpc-camera-guard is untouched).
pkill -f "python cam_server.py --source" 2>/dev/null
sleep 1
