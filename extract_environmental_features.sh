#!/bin/bash

function convert_to_grads_lat() {
    #TODO: bash doesn't support floating point arithmetic
    local deg=${1:0:$((${#1}-1))}
    deg=$(echo $deg | sed 's/^0*//')
    deg=$(($deg / 10))

    if [[ $1 == *N ]]; then
        echo $(($deg * -1))
    else
        echo $deg
    fi
}

function convert_to_grads_long() {
    #TODO: bash doesn't support floating point arithmetic
    local deg=${1:0:$((${#1}-1))}
    deg=$(echo $deg | sed 's/^0*//')
    deg=$(($deg / 10))

    if [[ $1 == *E ]]; then
        echo $((360 - $deg))
    else
        echo $deg
    fi
}

function cleanup_tmp_dir() {
    if [ -d "${TMP_DIR}" ]; then
        rm -rf "$TMP_DIR"
        echo "Temporary directory removed!"
    fi
}

# Make sure that the temporary directory is cleaned up even if the script fails.
trap cleanup_tmp_dir EXIT

# Source configuration file so we don't have to write
# long functions to parse command line arguments.
CONFIG_FILE=$1
if [ -z "${CONFIG_FILE}" ]; then
    echo "Please input the path of configuration file as first argument"
    exit 1
elif ! [ -f "${CONFIG_FILE}" ]; then
    echo "$CONFIG_FILE doesn't exist. Please make sure that configuration file path is correct."
    exit 1
else
    echo "Loading configuration file ..."
    . $CONFIG_FILE
    echo "Successfully load configuration file."
fi

# Check that BDECK_FILES variable is set.
if [ -z "${BDECK_FILES}" ]; then
    echo "Missing variable BDECK_FILES in configuration file."
    exit 1
fi

# Check that REANALYSIS_DIR variable is set.
if [ -z "${REANALYSIS_FILES}" ]; then
    echo "Missing variable REANALYSIS_FILES in configuration file."
    exit 1
fi

# Check that OUTPUT_DIR variable is set,
# if not, fallback to the default one.
if [ -z "${OUTPUT_DIR}" ]; then
    echo "WARN: Missing variable OUTPUT_DIR in configuration file. Fallback to default directory `output`"
    OUTPUT_DIR="./output"
fi
# Create output directory if it does not exist.
[ -d "${OUTPUT_DIR}" ] || mkdir -p $OUTPUT_DIR

# Create output file to store all tropical cyclones appearance.
TC_FILE="${OUTPUT_DIR}/tc.csv"
echo "Observation,Genesis,Latitude,Longitude" > "${TC_FILE}"

# Store time frame that we shouldn't extract environmental variables from.
# Because either we're having TCs in those time, or we want to extract environmental variables before the lead time.
TC_LEAD_TIME=()
TC_END_TIME=()

# Loop through all bdeck files to get tropical cyclones genesis date.
for bdeck_file in $BDECK_FILES; do
    first_line=$(head -n 1 $bdeck_file)
    last_line=$(tail -n 1 $bdeck_file)

    tc_genesis_time=$(echo $first_line | awk -F, '{print $3}')
    tc_end_time=$(echo $last_line | awk -F, '{print $3}')

    tc_year=$(echo $tc_genesis_time | cut -b 1-4)
    tc_month=$(echo $tc_genesis_time | cut -b 5-6)
    tc_day=$(echo $tc_genesis_time | cut -b 7-8)
    tc_hour=$(echo $tc_genesis_time | cut -b 9-10)

    # Calculate the time from which we should get observation data from.
    tc_observation_time=$(date --date="${tc_year}${tc_month}${tc_day} ${tc_hour} -${LEAD_TIME} hour" +"%Y%m%d %H" | sed -e 's/ //g')

    TC_LEAD_TIME+=($tc_observation_time)
    TC_END_TIME+=($tc_end_time)

    tc_lat=$(echo $first_line | awk -F, '{print $7}')
    tc_lat="$(convert_to_grads_lat $tc_lat)"
    tc_long=$(echo $first_line | awk -F, '{print $8}')
    tc_long="$(convert_to_grads_long $tc_long)"

    echo "${tc_observation_time},${tc_genesis_time},${tc_lat},${tc_long}" >> $TC_FILE
done

# Create a template Grads script to extract data from GRIB2 and convert it to intermediate netCDF files.
# The function uses global variables: LATITUDE, LONGITUDE, LEVELS, and VARIABLES
# $1: output file
function generate_grads_template_script() {
    GRIB_CTL_FILE="__GRIB_CTL_FILE__"
    GRIB_CTL_FILE_PREFIX="__GRIB_CTL_FILE_PREFIX__"

    # Generate statements to put in the template grads script.
    local statements=(
        "'reinit'"
        "'open ${GRIB_CTL_FILE}'"
        "'set lat ${LATITUDE[0]} ${LATITUDE[1]}'"
        "'set lon ${LONGITUDE[0]} ${LONGITUDE[1]}'"
        "'set lev ${LEVELS[0]}'"
        "'set t 1 last'"
    )
    

    for variable in "${VARIABLES[@]}"; do
        statements+=(
            "'define ${variable}=${variable}'"
            "'set sdfwrite -flt ${GRIB_CTL_FILE_PREFIX}.${variable}.nc'"
            "'sdfwrite ${variable}'"
        )
    done

    # Output all statement to template grads script.
    echo > $1
    for statement in "${statements[@]}"; do
        echo $statement >> $1
    done
}

# Generate a temporary directory to store all of our temporary Grads scripts.
TMP_DIR=$(mktemp -d)
if [ ! -e $TMP_DIR ]; then
    >&2 echo "ERROR: Failed to create temporary directory!"
    exit 1
fi
echo "Temporary directory ${TMP_DIR} created!"

# Function will generate netCDF from Grib2 observation data file.
# It will use the template grads script to generate intermediate netCDF files,
# then merge those files using `cdo` and then remove intermediate files.
# $1: path to the template grads script.
# $2: grib2 file to extract data from and convert to netCDF file.
# $3: path to directory contains the final netCDF file,
#   the filename will follow the original grib2, but substitute .gs with .nc
function generate_netcdf() {
    # Generate a unique prefix so we can easily clean up intermediate files.
    local unique_prefix=$(cat /proc/sys/kernel/random/uuid | sed 's/[-]//g' | head -c 20)

    # Check if .ctl already exists for the current .grib2 file,
    # if it doesn't create it.
    local ctl_file="${2}.ctl"
    [ -f "${ctl_file}" ] || ./3rd_party/g2ctl -0 "${2}" > "${ctl_file}"

    # Then, execute the grads script to generate intermediate netCDF files.
    sed "s|${GRIB_CTL_FILE}|${ctl_file}|g" "${1}" > $unique_prefix.gs
    sed -i "s|${GRIB_CTL_FILE_PREFIX}|${unique_prefix}|g" $unique_prefix.gs
    grads -xlbc $unique_prefix.gs > /dev/null

    # Merge all those netCDF files to one file.
    cdo -O -s merge $unique_prefix.*.nc "${3}/$(basename -- "$2").nc" > /dev/null

    # Remove all intermediate netCDF files and grads script.
    rm $unique_prefix.*.nc $unique_prefix.gs
}

# Making sure that we're dealing with absolute path,
# because we're going to change to temporary directory.
ABS_REANALYSIS_FILES=( $(readlink -f "${REANALYSIS_FILES[@]}") )
OUTPUT_DIR=$(readlink -f $OUTPUT_DIR)

# Generate template Grads script.
GRADS_EXTRACT_DATA_TEMPLATE="${TMP_DIR}/data_extract.template.gs"
generate_grads_template_script "${GRADS_EXTRACT_DATA_TEMPLATE}"

# Change to temporary directory and do all our works there,
# to avoid littering current directory.
pushd "${TMP_DIR}"

# Loop through all observation data to extract those values.
for data_file in "${ABS_REANALYSIS_FILES[@]}"; do
    # Get the date of the observation.
    observation_date=$(wgrib2 $data_file | head -n 1 | awk -F: '{print $3}' | cut -b 3-)

    # If the observation is in the periods that a tropical cyclone is happening,
    # we will skip this observation.
    skip=0;
    for i in "${!TC_LEAD_TIME[@]}"; do
        if [[ "$observation_date" > "${TC_LEAD_TIME[i]}" && "$observation_date" < "${TC_END_TIME[i]}" || "$observation_date" == "${TC_END_TIME[i]}" ]]; then
            skip=1
            break
        fi
    done

    # Only extract observation data when the date is valid.
    [ $skip -ne 0 ] || generate_netcdf "${GRADS_EXTRACT_DATA_TEMPLATE}" "${data_file}" "${OUTPUT_DIR}"
done

# After finish everything, return back to the original directory to do further stuffs.
popd
