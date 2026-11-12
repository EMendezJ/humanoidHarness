#!/bin/bash
# SAM2 segmentation server for the pick (CPU on the Jetson, ~4 s per mask). Runs on cores 3-7, away
# from the camera guard. Stop it with Ctrl+C.
cd "$(dirname "$0")"
exec taskset -c 3-7 ./hh.sh segment.py serve --device cpu --threads 4 "$@"
