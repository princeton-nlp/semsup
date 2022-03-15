#!/bin/bash

SEED=0
RUN_SCRIPT="../run.py"
CONFIG_FOLDER="./awa_configs"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/awa_scen1_clsname.yaml"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/awa_scen2_clsname.yaml"
)
ARGS="--train --default_config $CONFIG_FOLDER/awa_default.yaml --seed $SEED --name_suffix s$SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done