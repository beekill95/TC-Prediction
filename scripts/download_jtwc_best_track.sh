#!/bin/bash

OUTPUT_DIR=$1
mkdir -p "$OUTPUT_DIR"

# Only download from the year 2008 and 2020.
YEARS=$(seq 2008 1 2020)

for year in ${YEARS[@]}; do
    # Get the best track data from JTWC
    url="https://www.metoc.navy.mil/jtwc/products/best-tracks/$year/${year}s-bwp/bwp$year.zip"
    wget $url -P "$OUTPUT_DIR"

    # Create output directory to unzip.
    unzip "$OUTPUT_DIR/bwp$year.zip" -d "$OUTPUT_DIR"

    # Clean up the zip file.
    rm "$OUTPUT_DIR/bwp$year.zip"
done
