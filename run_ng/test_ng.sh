CONFIG_FOLDER="./ng_configs"
CKPT_FOLDER="../checkpoints_pt/newsgroups"
RUN_CMD="python ../run_eval.py --validate --default_config $CONFIG_FOLDER/ng_default.yaml --data.batch_size 5 --data.args.run_test true"

DEVISE="--ModelCls DEVISEBaseline --data.args.setup_glove_embeddings true"
GILE="--ModelCls DEVISEBaseline --model.args.use_gile true --data.args.setup_glove_embeddings true"

TRAIN="../class_descrs/newsgroups/combined_ng_manual_train.labels"
TRAIN_DESCR1="../class_descrs/newsgroups/combined_ng_manual_train_descr1.labels"
EVAL="../class_descrs/newsgroups/combined_ng_manual_test.labels"
CLSNAME="../class_descrs/newsgroups/ng_classnames.labels"
SUPER="../class_descrs/newsgroups/combined_ng_superclass_manual_test.labels"
SUPER_CLSNAME="../class_descrs/newsgroups/ng_superclass_classnames.labels"
SENTI="../class_descrs/newsgroups/combined_ng_manual_sentiment_test.labels"
SENTI_CLSNAME="../class_descrs/newsgroups/ng_sentiment_classname.labels"

echo
echo "TESTING NEWSGROUPS: SETTING 0"
echo

echo "Base"
$RUN_CMD --config "$CONFIG_FOLDER/ng_sup.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/base/"ng*.ckpt "$@"
echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/semlab/"ng_scen1*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" --data.args.val_label_json $TRAIN_DESCR1 --checkpoints "$CKPT_FOLDER/descr1/"ng_scen1*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/clsname/"ng_scen1_clsname_s*ckpt "$@"
echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" $DEVISE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/devise/"ng_scen1_clsname*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" $GILE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/gile/"ng_scen1_clsname*ckpt "$@"
echo "DEVISE (Descriptions)"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" $DEVISE --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/devise/"ng_scen1_devise*ckpt "$@"
echo "GILE (Descriptions)"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" $GILE --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/gile/"ng_scen1_gile*ckpt "$@"



echo
echo "TESTING NEWSGROUPS: SETTING 1"
echo

echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/semlab/"ng_scen1*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/descr1/"ng_scen1*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/clsname/"ng*NL*ckpt "$@"
echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" $DEVISE --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/devise/"ng_scen1_devise*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen1.yaml" $GILE --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/gile/"ng_scen1_gile*ckpt "$@"

echo
echo "TESTING NEWSGROUPS: SETTING 2"
echo

echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen2.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/semlab/"ng*heldout*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen2.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/descr1/"ng*heldout*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen2.yaml" --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/clsname/"ng*heldout*ckpt "$@"
echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen2.yaml" $DEVISE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/devise/"ng*heldout*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen2.yaml" $GILE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/gile/"ng*heldout*ckpt "$@"

echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen2.yaml" --data.args.val_label_json $TRAIN_DESCR1 --checkpoints "$CKPT_FOLDER/descr1/"ng*heldout*ckpt "$@"

echo
echo "TESTING NEWSGROUPS: SETTING 3"
echo

echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen3.yaml" --data.args.val_label_json $SUPER --checkpoints "$CKPT_FOLDER/semlab/"ng*super*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen3.yaml" --data.args.val_label_json $SUPER --checkpoints "$CKPT_FOLDER/descr1/"ng*super*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen3.yaml" --data.args.val_label_json $SUPER_CLSNAME --checkpoints "$CKPT_FOLDER/clsname/"ng*super*ckpt "$@"
echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen3.yaml" $DEVISE --data.args.val_label_json $SUPER_CLSNAME --checkpoints "$CKPT_FOLDER/devise/"ng*super*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_scen3.yaml" $GILE --data.args.val_label_json $SUPER_CLSNAME --checkpoints "$CKPT_FOLDER/gile/"ng*super*ckpt "$@"
