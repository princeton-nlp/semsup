CONFIG_FOLDER="./awa_configs"
CKPT_FOLDER="../checkpoints_pt/awa"
RUN_CMD="python ../run_eval.py --validate --default_config $CONFIG_FOLDER/awa_default.yaml --data.batch_size 64 --data.args.run_test true --data.args.remove_cifar true"

DEVISE="--ModelCls DEVISEVisBaseline --data.args.setup_glove_embeddings true"
GILE="--ModelCls DEVISEVisBaseline --model.args.use_gile true --data.args.setup_glove_embeddings true"

TRAIN="../class_descrs/awa/large_files/awa_deep_samp50_perm25_train.labels"
TRAIN_NL="../class_descrs/awa/google_awa_manual_train.labels"
TRAIN_DESCR1="../class_descrs/awa/awa_base_deep.labels"
TRAIN_NL_DESCR1="../class_descrs/awa/google_awa_manual_train_descr1.labels"
EVAL="../class_descrs/awa/large_files/awa_deep_samp50_perm25_test.labels"
EVAL_NL="../class_descrs/awa/google_awa_manual_test.labels"
CLSNAME="../class_descrs/awa/awa_clsnames.labels"
ATTRLIST="../class_descrs/awa/awa_attrlist.labels"
TRAIN_BOV="../class_descrs/awa/large_files/awa_deep_samp50_perm25_train_bov.labels"
EVAL_BOV="../class_descrs/awa/large_files/awa_deep_samp50_perm25_test_bov.labels"

echo
echo "TESTING AWA: SETTING 0"
echo

echo "Base"
$RUN_CMD --config "$CONFIG_FOLDER/awa_sup.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/base/"awa*.ckpt "$@"
echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/semlab/"awa*cc*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" --data.args.val_label_json $TRAIN_DESCR1 --checkpoints "$CKPT_FOLDER/descr1/"awa*cc*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/clsname/"*clsname_s*ckpt "$@"

echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" $DEVISE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/devise/"*devise_s*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" $GILE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/gile/"*gile_s*ckpt "$@"
echo "DEVISE (Descriptions)"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1_bov.yaml" $DEVISE --data.args.val_label_json $TRAIN_BOV --checkpoints "$CKPT_FOLDER/devise/"awa*bov*_s*ckpt "$@"
echo "GILE (Descriptions)"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1_bov.yaml" $GILE --data.args.val_label_json $TRAIN_BOV --checkpoints "$CKPT_FOLDER/gile/"awa*bov*_s*ckpt "$@"


echo
echo "TESTING AWA: SETTING 1"
echo

echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/semlab/"awa*cc*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/descr1/"awa*cc*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/clsname/"*NL*ckpt "$@"

echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" $DEVISE --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/devise/"*NL*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1.yaml" $GILE --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/gile/"*NL*ckpt "$@"
echo "DEVISE (Descriptions)"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1_bov.yaml" $DEVISE --data.args.val_label_json $EVAL_BOV --checkpoints "$CKPT_FOLDER/devise/"awa*bov*_s*ckpt "$@"
echo "GILE (Descriptions)"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen1_bov.yaml" $GILE --data.args.val_label_json $EVAL_BOV --checkpoints "$CKPT_FOLDER/gile/"awa*bov*_s*ckpt "$@"


echo
echo "TESTING AWA: SETTING 2"
echo

echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen2.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/semlab/"awa_scen2*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen2.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/descr1/"awa_scen2*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen2_clsname.yaml" --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/clsname/"awa_scen2*ckpt "$@"

echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen2_clsname.yaml" $DEVISE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/devise/"awa_scen2*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen2_clsname.yaml" $GILE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/gile/"awa_scen2*ckpt "$@"

echo "SemLab NL"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen2_nl.yaml" --data.args.val_label_json $TRAIN_NL --checkpoints "$CKPT_FOLDER/semlab/"awa*nl_heldout_s*ckpt "$@"
echo "SemLab NL Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/awa_scen2_nl.yaml" --data.args.val_label_json $TRAIN_NL --checkpoints "$CKPT_FOLDER/descr1/"awa*nl_heldout_descr1*ckpt "$@"