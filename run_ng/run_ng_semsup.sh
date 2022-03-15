#!/bin/bash

SEED=1
CONFIG_FOLDER="./ng_configs"
RUN_SCRIPT="../run.py"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen1.yaml"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen2.yaml"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen3.yaml"
)
ARGS="--train --default_config $CONFIG_FOLDER/ng_default.yaml --seed $SEED --name_suffix s$SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done