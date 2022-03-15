#!/bin/bash

SEED=1
CONFIG_FOLDER="./ng_configs"
RUN_SCRIPT="../run.py"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen1_clsname.yaml --name_suffix s$SEED"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen1_clsname.yaml --data.args.val_label_json ../class_descrs/newsgroups/combined_ng_manual_val.labels --name_suffix NL_s$SEED"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen2_clsname.yaml --name_suffix s$SEED"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen3_clsname.yaml --name_suffix s$SEED"
)
ARGS="--train --default_config $CONFIG_FOLDER/ng_default.yaml --seed $SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done