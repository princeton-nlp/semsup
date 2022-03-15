#!/bin/bash

SEED=1
RUN_SCRIPT="../run.py"
CONFIG_FOLDER="./cifar_configs"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_scen1_clsname.yaml --name_suffix s$SEED"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_scen1_clsname.yaml --data.args.val_label_json ../class_descrs/cifar/combined_cifar100_manual_val.labels --name_suffix NL_s$SEED"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_scen2_clsname.yaml --name_suffix s$SEED"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_scen3_clsname.yaml --name_suffix s$SEED"
)
ARGS="--train --default_config $CONFIG_FOLDER/cifar_default.yaml --seed $SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done