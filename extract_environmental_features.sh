#!/bin/bash

# UTILITIES FUNCTIONS
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
        "'set lev ${LEVELS[0]}'"
        "'set t 1 last'"
    )
    

    for variable in "${VARIABLES[@]}"; do
        statements+=(
            "'define ${variable}=${variable}'"
            "'set sdfwrite -flt ${3}.${variable}.nc'"
            "'sdfwrite ${variable}'"
        )
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
function generate_netcdf() {
    # Generate a unique prefix so we can easily clean up intermediate files.
    local unique_prefix=$(cat /proc/sys/kernel/random/uuid | sed 's/[-]//g' | head -c 20)

    # Check if .ctl already exists for the current .grib2 file,
    # if it doesn't create it.
    local ctl_file="${1}.ctl"
    [ -f "${ctl_file}" ] || ./3rd_party/g2ctl -0 "${1}" > "${ctl_file}"

    # Then, execute the grads script to generate intermediate netCDF files.
    generate_grads_extract_script "${unique_prefix}.gs" "$1.ctl" $unique_prefix
    grads -xlbc "${unique_prefix}.gs" > /dev/null

    # Merge all those netCDF files to one file.
    cdo -O -s merge $unique_prefix.*.nc "${2}/$(basename -- "$1" .grib2).nc" > /dev/null

    # Remove all intermediate netCDF files and grads script.
    rm $unique_prefix.*.nc $unique_prefix.gs
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
# Create a directory in the output directory for pressure level.
OUTPUT_DIR="$OUTPUT_DIR/${LEVELS[0]}mb"
# Create output directory if it does not exist.
[ -d "${OUTPUT_DIR}" ] || mkdir -p "$OUTPUT_DIR"

# Store time frame that we shouldn't extract environmental variables from.
# Because either we're having TCs in those time, or we want to extract environmental variables before the lead time.
TC_LEAD_TIME=()
TC_GENESIS_TIME=()
TC_END_TIME=()
TC_LAT=()
TC_LONG=()

# Loop through all bdeck files to get tropical cyclones genesis date.
for bdeck_file in "${BDECK_FILES[@]}"; do
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

    tc_lat=$(echo $first_line | awk -F, '{print $7}')
    tc_lat="$(convert_to_grads_lat $tc_lat)"
    tc_long=$(echo $first_line | awk -F, '{print $8}')
    tc_long="$(convert_to_grads_long $tc_long)"

    TC_LEAD_TIME+=($tc_observation_time)
    TC_GENESIS_TIME+=($tc_genesis_time)
    TC_END_TIME+=($tc_end_time)
    TC_LAT+=($tc_lat)
    TC_LONG+=($tc_long)
    # echo "${tc_observation_time},${tc_genesis_time},${tc_end_time},${tc_lat},${tc_long}" >> $TC_FILE
done

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

NO_TC_OBSERVATIONS=()

# Loop through all observation data to extract those values.
for data_file in "${ABS_REANALYSIS_FILES[@]}"; do
    # Get the date of the observation.
    observation_date=$(wgrib2 $data_file | head -n 1 | awk -F: '{print $3}' | cut -b 3-)

    echo "Processing ${data_file}!"

    # If the observation is in the periods that a tropical cyclone is happening,
    # we will skip this observation.
    skip=0
    for i in "${!TC_LEAD_TIME[@]}"; do
        if [[ "$observation_date" > "${TC_LEAD_TIME[i]}" && "$observation_date" < "${TC_END_TIME[i]}" || "$observation_date" == "${TC_END_TIME[i]}" ]]; then
            echo "Skipped $observation_date"
            skip=1
            break
        fi
    done


    # Only extract observation data when the date is valid.
    if [ $skip -eq 0 ]; then
        echo "Generating netCDF from ${data_file}!"

        generate_netcdf "${data_file}" "${OUTPUT_DIR}"

        # To store observation files where we don't found any tropical cyclones.
        if ! [[ "${TC_LEAD_TIME[*]}" == *"$observation_date"* ]]; then
            NO_TC_OBSERVATIONS+=($observation_date)
        fi
    fi

    echo "Done processing ${data_file}!"
done

# After finish everything, return back to the original directory to do further stuffs.
popd

# Copy the configuration file to the output directory so we know how the data was extracted.
cp $1 "${OUTPUT_DIR}/conf"
chmod u-wx "${OUTPUT_DIR}/conf"

# Create output file to store label of all observation files.
TC_FILE="${OUTPUT_DIR}/tc.csv"
echo "Observation,TC,Genesis,End,Latitude,Longitude" > "${TC_FILE}"

for idate in "${NO_TC_OBSERVATIONS[@]}"; do
    echo "$idate,0,,,,," >> "${TC_FILE}"
done

for i in "${!TC_LEAD_TIME[@]}"; do
    echo "${TC_LEAD_TIME[i]},1,${TC_GENESIS_TIME[i]},${TC_END_TIME[i]},${TC_LAT[i]},${TC_LONG[i]}" >> "${TC_FILE}"
done

# Then, sort the final output file.
{ head -n 1 "${TC_FILE}" & tail -n +2 "${TC_FILE}" | sort -t , -k1,1; } > "${TC_FILE}"
