#!/bin/bash

SEED=0
RUN_SCRIPT="../run.py"
CONFIG_FOLDER="./configs/awa_configs"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/awa_sup.yaml"
)
ARGS="--train --default_config $CONFIG_FOLDER/awa_default.yaml --seed $SEED --name_suffix s$SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done