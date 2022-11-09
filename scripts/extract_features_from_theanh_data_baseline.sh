#!/bin/bash

# This is the files that should be processed by the script.
WRF_FILES=(/N/project/pfec_climo/anhvu/WRFV3/WPAC_baseline_18km/WPAC_baseline_????/raw_wrfout_d01_*)
OUTPUT_DIR="data/theanh_WPAC_baseline_3/"
OUTPUT_PREFIX="baseline"

# Execute the task.
./scripts/extract_features_from_theanh_data.py \
    "$OUTPUT_DIR" "${WRF_FILES[@]}"  \
    --prefix "$OUTPUT_PREFIX"

# We're done.
echo '=== DONE!!! ==='
