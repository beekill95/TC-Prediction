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

# Source configuration file so we don't have to write
# long functions to parse command line arguments.
CONFIG_FILE=$1
if [ -z $CONFIG_FILE ]; then
    echo "Please input the path of configuration file as first argument"
    exit 1
elif ! [ -f $CONFIG_FILE ]; then
    echo "$CONFIG_FILE doesn't exist. Please make sure that configuration file path is correct."
    exit 1
else
    echo "Loading configuration file ..."
    . $CONFIG_FILE
    echo "Successfully load configuration file."
fi

# Check that BDECK_FILES variable is set.
if [ -z $BDECK_FILES ]; then
    echo "Missing variable BDECK_FILES in configuration file."
    exit 1
fi

# Check that REANALYSIS_DIR variable is set.
if [ -z $REANALYSIS_DIR ]; then
    echo "Missing variable REANALYSIS_DIR in configuration file."
    exit 1
fi

# Check that OUTPUT_DIR variable is set,
# if not, fallback to the default one.
if [ -z $OUTPUT_DIR ]; then
    echo "WARN: Missing variable OUTPUT_DIR in configuration file. Fallback to default directory `output`"
    OUTPUT_DIR="./output"
fi
# Create output directory if it does not exist.
[-d $OUTPUT_DIR ] || mkdir -p $OUTPUT_DIR

for bdeck_file in $BDECK_FILES; do
    first_line=$(head -n 1 $bdeck_file)

    tc_genesis_time=$(echo $first_line | awk -F, '{print $3}')
    tc_year=$(echo $tc_genesis_time | cut -b 1-4)
    tc_month=$(echo $tc_genesis_time | cut -b 5-6)
    tc_day=$(echo $tc_genesis_time | cut -b 7-8)
    tc_hour=$(echo $tc_genesis_time | cut -b 9-10)

    tc_lat=$(echo $first_line | awk -F, '{print $7}')
    tc_lat="$(convert_to_grads_lat $tc_lat)"
    tc_long=$(echo $first_line | awk -F, '{print $8}')
    tc_long="$(convert_to_grads_long $tc_long)"

    # echo $tc_genesis_time
    echo $tc_lat
    echo $tc_long
done

