#!/bin/bash

SEED=1
CONFIG_FOLDER="./ng_configs"
RUN_SCRIPT="../run.py"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen1.yaml"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen1_clsname.yaml"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen2_clsname.yaml"
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/ng_scen3_clsname.yaml"
)

# gile args
ARGS="--train --default_config $CONFIG_FOLDER/ng_default.yaml --ModelCls DEVISEBaseline --model.args.use_gile true --data.args.setup_glove_embeddings true --seed $SEED --name_suffix gile_s$SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done