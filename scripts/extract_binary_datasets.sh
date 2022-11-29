#!/bin/bash

leadtimes=(0 6 12 24 36 48 60 72)

# Extract The Anh's baseline
# for leadtime in "${leadtimes[@]}"; do
#     echo "Creating From The Anh's Baseline ${leadtime}h"
#     scripts/create_tc_binary_classification_dataset_theanh.py \
#         --best-track "/N/project/pfec_climo/anhvu/Research/Tracking_code/18km/tccount_final_baseline_????.txt" \
#         --theanh-baseline data/theanh_WPAC_baseline_5 \
#         --distances 50 40 30 \
#         --leadtime $leadtime \
#         --output "/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/WRF_baseline_5_binary_${leadtime}h"
# done

# Extract The Anh's RCP45
for leadtime in "${leadtimes[@]}"; do
    echo "Creating From The Anh's RCP45 ${leadtime}h"
    scripts/create_tc_binary_classification_dataset_theanh.py \
        --best-track "/N/slate/anhvu/Tracking_code/9km/WPAC_RCP45_2*/tccount_final.txt" \
        --theanh-baseline data/theanh_WPAC_RCP45_5 \
        --distances 50 40 30 \
        --leadtime $leadtime \
        --output "/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/WRF_RCP45_5_binary_${leadtime}h"
done

# Extract The Anh's RCP85
for leadtime in "${leadtimes[@]}"; do
    echo "Creating From The Anh's RCP85 ${leadtime}h"
    scripts/create_tc_binary_classification_dataset_theanh.py \
        --best-track "/N/slate/anhvu/Tracking_code/9km/WPAC_RCP85_2*/tccount_final.txt" \
        --theanh-baseline data/theanh_WPAC_RCP85_5 \
        --distances 50 40 30 \
        --leadtime $leadtime \
        --output "/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/WRF_RCP85_5_binary_${leadtime}h"
done

# Extract From NCEP/FNL WP
# for leadtime in "${leadtimes[@]}"; do
#     echo "Creating From NCEP/FNL WP ${leadtime}h"
#     scripts/create_tc_binary_classification_ncep.py \
#         --best-track ibtracs.ALL.list.v04r00.csv \
#         --ncep-fnl /N/project/pfec_climo/qmnguyen/tc_prediction/data/ncep_fnl \
#         --leadtime $leadtime \
#         --basin WP \
#         --output "/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/ncep_WP_binary_${leadtime}h"
# done

# Extract From NCEP/FNL EP
# for leadtime in "${leadtimes[@]}"; do
#     echo "Creating From NCEP/FNL EP ${leadtime}h"
#     scripts/create_tc_binary_classification_ncep.py \
#         --best-track ibtracs.ALL.list.v04r00.csv \
#         --ncep-fnl /N/project/pfec_climo/qmnguyen/tc_prediction/data/ncep_fnl \
#         --leadtime $leadtime \
#         --basin EP \
#         --output "/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/ncep_EP_binary_${leadtime}h"
# done


# Extract From NCEP/FNL NA
# for leadtime in "${leadtimes[@]}"; do
#     echo "Creating From NCEP/FNL NA ${leadtime}h"
#     scripts/create_tc_binary_classification_ncep.py \
#         --best-track ibtracs.ALL.list.v04r00.csv \
#         --ncep-fnl /N/project/pfec_climo/qmnguyen/tc_prediction/data/ncep_fnl \
#         --leadtime $leadtime \
#         --basin NA \
#         --output "/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/ncep_NA_binary_${leadtime}h"
# done
