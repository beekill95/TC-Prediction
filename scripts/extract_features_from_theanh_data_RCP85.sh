#!/bin/bash

# This is the files that should be processed by the script.
WRF_FILES="/N/project/pfec_climo/anhvu/WRFV3/WPAC_RCP85_2???/raw_wrfout_d01_*"
OUTPUT_DIR="data/theanh_WPAC_RCP85_5/"
OUTPUT_PREFIX="RCP85"

# Execute the task.
echo '=== Running!!! ==='
./scripts/extract_features_from_theanh_data.py \
    "$OUTPUT_DIR" "$WRF_FILES"  \
    --prefix "$OUTPUT_PREFIX" \
    --cores 32

# We're done.
echo '=== DONE!!! ==='
