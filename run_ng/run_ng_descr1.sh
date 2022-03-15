#!/bin/bash

SEED=1
CONFIG_FOLDER="./ng_configs"
RUN_SCRIPT="../run.py"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen1.yaml --data.args.train_label_json ../class_descrs/newsgroups/combined_ng_manual_train_descr1.labels"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen2.yaml --data.args.train_label_json ../class_descrs/newsgroups/combined_ng_manual_train_descr1.labels"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen3.yaml --data.args.train_label_json ../class_descrs/newsgroups/combined_ng_manual_train_descr1.labels"
)
ARGS="--train --default_config $CONFIG_FOLDER/ng_default.yaml --seed $SEED --name_suffix descr1_s$SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done