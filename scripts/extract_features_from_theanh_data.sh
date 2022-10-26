#!/bin/bash

# This is the files that should be processed by the script.
# WRF_FILES=(/N/project/pfec_climo/anhvu/WRFV3/WPAC_baseline_18km/WPAC_baseline_????/raw_wrfout_d01_*)
# OUTPUT_DIR="data/theanh_WPAC_baseline_2/"
# OUTPUT_PREFIX="baseline"

# WRF_FILES=(/N/project/pfec_climo/anhvu/WRFV3/WPAC_RCP45_2???/raw_wrfout_d01_*)
# OUTPUT_DIR="data/theanh_WPAC_RCP45_2/"
# OUTPUT_PREFIX="RCP45"

WRF_FILES=(/N/project/pfec_climo/anhvu/WRFV3/WPAC_RCP85_2???/raw_wrfout_d01_*)
OUTPUT_DIR="data/theanh_WPAC_RCP85_2/"
OUTPUT_PREFIX="RCP85"

# Execute the task.
# find /N/project/pfec_climo/anhvu/WRFV3/WPAC_RCP85_2???/ \
#     -prune -name 'raw_wrfout_d01_*' \
#     -exec ./scripts/extract_features_from_theanh_data.py "$OUTPUT_DIR" {} --prefix "$OUTPUT_PREFIX" +
./scripts/extract_features_from_theanh_data.py \
    "$OUTPUT_DIR" "${WRF_FILES[@]}"  \
    --prefix "$OUTPUT_PREFIX"

# We're done.
echo '=== DONE!!! ==='
