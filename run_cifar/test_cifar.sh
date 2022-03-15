CONFIG_FOLDER="./cifar_configs"
CKPT_FOLDER="../checkpoints_pt/cifar"
RUN_CMD="python ../run_eval.py --validate --default_config $CONFIG_FOLDER/cifar_default.yaml --data.batch_size 5 --data.args.run_test true"

DEVISE="--ModelCls DEVISEVisBaseline --data.args.setup_glove_embeddings true"
GILE="--ModelCls DEVISEVisBaseline --model.args.use_gile true --data.args.setup_glove_embeddings true"

TRAIN="../class_descrs/cifar/combined_cifar100_manual_train.labels"
TRAIN_DESCR1="../class_descrs/cifar/combined_cifar100_manual_train_descr1.labels"
EVAL="../class_descrs/cifar/combined_cifar100_manual_test.labels"
CLSNAME="../class_descrs/cifar/cifar_classnames_notemplate.labels"
SUPER="../class_descrs/cifar/google_cifar_super_manual_test.labels"
SUPER_CLSNAME="../class_descrs/cifar/cifar_super_classnames_notemplate.labels"


echo
echo "TESTING CIFAR: SETTING 0"
echo

BASE EXPERIMENTS
echo "Base"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_sup.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/base/"cifar*.ckpt "$@"
echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/semlab/"cifar*cc*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" --data.args.val_label_json $TRAIN_DESCR1 --checkpoints "$CKPT_FOLDER/descr1/"cifar*cc*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/clsname/"cifar*cc*clsname_s*ckpt "$@"
echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" $DEVISE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/devise/"cifar*cc*clsname_devise_s*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" $GILE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/gile/"cifar*cc*clsname_gile_s*ckpt "$@"
echo "DEVISE (Descriptions)"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" $DEVISE --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/devise/"cifar*cc_devise*ckpt "$@"
echo "GILE (Descriptions)"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" $GILE --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/gile/"cifar*cc_gile*ckpt "$@"

echo
echo "TESTING CIFAR: SETTING 1"
echo

echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/semlab/"cifar*cc*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/descr1/"cifar*cc*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/clsname/"cifar*NL*ckpt "$@"
echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" $DEVISE --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/devise/"cifar*cc_devise*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen1.yaml" $GILE --data.args.val_label_json $EVAL --checkpoints "$CKPT_FOLDER/gile/"cifar*cc_gile*ckpt "$@"

echo
echo "TESTING CIFAR: SETTING 2"
echo

echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen2.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/semlab/"cifar*heldout*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen2.yaml" --data.args.val_label_json $TRAIN --checkpoints "$CKPT_FOLDER/descr1/"cifar*heldout*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen2.yaml" --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/clsname/"cifar*heldout*ckpt "$@"
echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen2.yaml" $DEVISE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/devise/"cifar*heldout*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen2.yaml" $GILE --data.args.val_label_json $CLSNAME --checkpoints "$CKPT_FOLDER/gile/"cifar*heldout*ckpt "$@"

echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen2.yaml" --data.args.val_label_json $TRAIN_DESCR1 --checkpoints "$CKPT_FOLDER/descr1/"cifar*heldout*ckpt "$@"

echo
echo "TESTING CIFAR: SETTING 3"
echo

echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen3.yaml" --data.args.val_label_json $SUPER --checkpoints "$CKPT_FOLDER/semlab/"cifar*super*ckpt "$@"
echo "SemLab Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen3.yaml" --data.args.val_label_json $SUPER --checkpoints "$CKPT_FOLDER/descr1/"cifar*super*ckpt "$@"
echo "SemLab classnames"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen3.yaml" --data.args.val_label_json $SUPER_CLSNAME --checkpoints "$CKPT_FOLDER/clsname/"cifar*super*ckpt "$@"
echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen3.yaml" $DEVISE --data.args.val_label_json $SUPER_CLSNAME --checkpoints "$CKPT_FOLDER/devise/"cifar*super*ckpt "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/cifar_scen3.yaml" $GILE --data.args.val_label_json $SUPER_CLSNAME --checkpoints "$CKPT_FOLDER/gile/"cifar*super*ckpt "$@"
