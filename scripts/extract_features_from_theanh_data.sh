#!/bin/bash

# This is the files that should be processed by the script.
WRF_FILES=(/N/project/pfec_climo/anhvu/WRFV3/WPAC_baseline_18km/WPAC_baseline_????/raw_wrfout_d01_*)
OUTPUT_DIR="data/theanh_WPAC_baseline/"
OUTPUT_PREFIX="baseline"

# Number of parallel runs.
N_PROCESSES=16

# Now, loop each file and process it with python script.
for file in "${WRF_FILES[@]}"; do
    # Make sure we don't use more processes than allowed.
    ((i=i%N_PROCESSES)); ((i++==0)) && wait

    # Process the file.
    echo "Processing file $file"
    ./scripts/extract_features_from_theanh_data.py \
        "$file" "$OUTPUT_DIR" \
        --prefix "$OUTPUT_PREFIX" &
done

# We're done.
echo '=== DONE!!! ==='
