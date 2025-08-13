#!/bin/bash

SCRIPT="./scripts/mpc_planner.py"
MAP_NAME="map414diffuser"
RITP_FILENAME="randSddrx-414test-static-v2-dense-light"
MODE="dynamic"
# MODE="static"
PARALLEL_NUM=12

python "$SCRIPT" \
    "--map_name" "$MAP_NAME" \
    "--ritp_filename" "$RITP_FILENAME" \
    "--mode" "$MODE" \
    "--parallel_num" "$PARALLEL_NUM"
