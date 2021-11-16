#!/bin/bash

# UTILITIES FUNCTIONS

# Create a lock in the output folder,
# so to prevent two scripts output to the same directory.
# $1: output directory.
function lock_output_folder() {
    FOLDER_LOCK="$1/.lock"
    if { set -C; 2>/dev/null > "$FOLDER_LOCK"; }; then
        trap "rm -f $FOLDER_LOCK" EXIT
    else
        echo "There seems to be an another script running,"
        echo "and output to the same directory!!!"
        echo "Please change the config file to output to a different directory."
        exit 1
    fi
}

function cleanup_tmp_dir() {
    if [ -d "${TMP_DIR}" ]; then
        rm -rf "$TMP_DIR"
        echo "Temporary directory ${TMP_DIR} removed!"
    fi
}

# Create a Grads script to extract data from GRIB2 and convert it to intermediate netCDF files.
# The function uses global variables: LATITUDE, LONGITUDE, LEVELS, and VARIABLES
# $1: output file
# $2: GRIB2 .ctl file to read from.
# $3: unique prefix to separate the output netCDF files of this script.
function generate_grads_extract_script() {
    # Generate statements to put in the template grads script.
    local statements=(
        "'reinit'"
        "'open ${2}'"
        "'set lat ${LATITUDE[0]} ${LATITUDE[1]}'"
        "'set lon ${LONGITUDE[0]} ${LONGITUDE[1]}'"
        "'set t 1 last'"
    )
    

    for variable in "${!VARIABLES_PRESSURES[@]}"; do
        local pressures=${VARIABLES_PRESSURES[$variable]}
        local use_pressure_ranges=false
        if [[ "$pressures" == *"-"* ]]; then
            use_pressure_ranges=true
            local max_pressure="${pressures%%-*}"
            local min_pressure="${pressures##*-}"
        else
            local max_pressure=$(sort -n <<< "${pressures// /$'\n'}" | tail -n 1)
            local min_pressure=$(sort -n <<< "${pressures// /$'\n'}" | head -n 1)
        fi

        local nc_output="${3}.${variable}.nc"

        # If the we have multiple pressure levels,
        # extract from maximum to minimum.
        # Then, use `cdo` to select the one we're interested in.
        if [[ "$max_pressure" != "$min_pressure" ]]; then
            statements+=("'set lev $max_pressure $min_pressure'")
        else
            statements+=("'set lev $max_pressure'")
        fi

        statements+=(
            "'define $variable=${variable}'"
            "'set sdfwrite -flt $nc_output'"
            "'sdfwrite $variable'"
        )

        # Only select pressure levels we want if there are individual pressure levels,
        # i.e. when there are multiple pressure levels, and no pressure ranges.
        if [[ "$max_pressure" != "$min_pressure" ]] && [ $use_pressure_ranges = false ]; then
            statements+=("'!cdo -O sellevel,${pressures// /,} $nc_output $nc_output'")
        fi
    done

    # Output all statement to template grads script.
    echo > $1
    for statement in "${statements[@]}"; do
        echo $statement >> "$1"
    done
}

# Function will generate netCDF from Grib2 observation data file.
# It will use the template grads script to generate intermediate netCDF files,
# then merge those files using `cdo` and then remove intermediate files.
# $1: grib2 file to extract data from and convert to netCDF file.
# $2: path to directory contains the final netCDF file,
#   the filename will follow the original grib2, but substitute .gs with .nc
G2CTL_PATH=$(readlink -f "./3rd_party/g2ctl")
function generate_netcdf() {
    echo "### GENERATING netCDF for $1"
    # Generate a unique prefix so we can easily clean up intermediate files.
    local unique_prefix=$(cat /proc/sys/kernel/random/uuid | sed 's/[-]//g' | head -c 20)

    # Check if .ctl already exists for the current .grib2 file,
    # if it doesn't create it.
    local ctl_file="${1}.ctl"
    local idx_file="${1}.idx"
    [ -f "${ctl_file}" ] || $G2CTL_PATH -0 "${1}" > "${ctl_file}"
    [ -f "${idx_file}" ] || gribmap -i "${ctl_file}"

    # Then, execute the grads script to generate intermediate netCDF files.
    generate_grads_extract_script "${unique_prefix}.gs" "$1.ctl" $unique_prefix
    grads -xlbc "${unique_prefix}.gs"

    # Merge all those netCDF files to one file.
    cdo -O -s merge $unique_prefix.*.nc "${2}/$(basename -- "$1" .grib2).nc"

    # Remove all intermediate netCDF files and grads script.
    rm $unique_prefix.*.nc $unique_prefix.gs
    echo "### DONE generating netCDF for $1"
}

# #### SCRIPT START ####

# Make sure that the temporary directory is cleaned up even if the script fails.
trap cleanup_tmp_dir EXIT

# Source configuration file so we don't have to write
# long functions to parse command line arguments.
CONFIG_FILE=$1
if [ -z "${CONFIG_FILE}" ]; then
    >&2 echo "Please input the path of configuration file as first argument"
    exit 1
elif ! [ -f "${CONFIG_FILE}" ]; then
    >&2 echo "$CONFIG_FILE doesn't exist. Please make sure that configuration file path is correct."
    exit 1
else
    echo "Loading configuration file ..."
    . $CONFIG_FILE
    echo "Successfully load configuration file."
fi

# Check that BDECK_FILES variable is set.
if [ -z "${BDECK_FILES}" ]; then
    >&2 echo "Missing variable BDECK_FILES in configuration file."
    exit 1
fi

# Check that REANALYSIS_DIR variable is set.
if [ -z "${REANALYSIS_FILES}" ]; then
    >&2 echo "Missing variable REANALYSIS_FILES in configuration file."
    exit 1
fi

# Check that our dependencies are satisfied.
for prog in "grads" "cdo" "wgrib2"; do
    if ! [ -x "$(command -v $prog)" ]; then
        >&2 echo "ERROR: Missing program $prog!!"
        exit 1
    fi
done

# Check that OUTPUT_DIR variable is set,
# if not, fallback to the default one.
if [ -z "${OUTPUT_DIR}" ]; then
    echo "WARN: Missing variable OUTPUT_DIR in configuration file. Fallback to default directory `output`"
    OUTPUT_DIR="./output"
fi
# Create a directory in the output directory for different lead time.
OUTPUT_DIR="$OUTPUT_DIR/${LEAD_TIME}h"
# Create output directory if it does not exist.
[ -d "${OUTPUT_DIR}" ] || mkdir -p "$OUTPUT_DIR"

OUTPUT_CONF="${OUTPUT_DIR}/conf"
if [ -z "$OUTPUT_CONF" ]; then
    if diff "$CONFIG_FILE" "$OUTPUT_CONF" > /dev/null; then
        echo "The output config file is identical to your given config file."
        echo "This is an indication that the folder is already populated with previous run of this script with the same config!"
    else
        echo "The output folder already contains a config file from the previous file."
        echo "And it is different from the given config file."
        echo "Rerun the script will make the data inconsistent, which might cause error down the line."
    fi

    echo "Therefore, the script won't run!"
    echo "If you want to run the script anyway, remove the output directory!"
    exit 1
fi

# Lock the folder so nobody but us can output to this folder.
# This will also fail if there already a lock file there!
# And it will automatically unlock the folder once the script finishes.
lock_output_folder "$OUTPUT_DIR"

# Generate a temporary directory to store all of our temporary Grads scripts.
TMP_DIR=$(mktemp -d)
if [ ! -e $TMP_DIR ]; then
    >&2 echo "ERROR: Failed to create temporary directory!"
    exit 1
fi
echo "Temporary directory ${TMP_DIR} created!"

# Making sure that we're dealing with absolute path,
# because we're going to change to temporary directory.
ABS_REANALYSIS_FILES=( $(readlink -f "${REANALYSIS_FILES[@]}") )
OUTPUT_DIR=$(readlink -f $OUTPUT_DIR)

# Change to temporary directory and do all our works there,
# to avoid littering current directory.
pushd "${TMP_DIR}"

# Loop through all observation data to extract those values.
N=4
for data_file in "${ABS_REANALYSIS_FILES[@]}"; do
    month=$(basename "$data_file" | cut -b 9-10)

    ((i=i%N)); ((i++==0)) && wait

    if [[ $month > "04" ]] && [[ $month < "12" ]]; then
        generate_netcdf "${data_file}" "${OUTPUT_DIR}" &
    fi
done

# After finish everything, return back to the original directory to do further stuffs.
popd

# Copy the configuration file to the output directory so we know how the data was extracted.
cp $1 "$OUTPUT_CONF"
chmod u-wx "$OUTPUT_CONF"
