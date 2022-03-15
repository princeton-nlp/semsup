#!/bin/bash

SEED=1
RUN_SCRIPT="../run.py"
CONFIG_FOLDER="./cifar_configs"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_scen1.yaml"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_scen2.yaml"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_scen3.yaml"
)
ARGS="--train --default_config $CONFIG_FOLDER/cifar_default.yaml --data.args.train_label_json ../class_descrs/cifar/combined_cifar100_manual_train_descr1.labels --seed $SEED --name_suffix descr1_s$SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done