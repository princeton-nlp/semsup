CONFIG_FOLDER="./ng_configs"
CKPT_FOLDER=""
RUN_CMD="python ../run_eval.py --validate --default_config $CONFIG_FOLDER/ng_transfer.yaml --data.batch_size 5 --data.args.run_test true"

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
echo "TRANSFER RCV1 to NG"
echo

echo "DEVISE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_transfer.yaml" --ModelCls BertRCV1toNGDEVISE --data.args.setup_glove_embeddings true --model.args.checkpoint $CKPT_FOLDER/devise --data.args.val_label_json $CLSNAME "$@"
echo "GILE"
$RUN_CMD --config "$CONFIG_FOLDER/ng_transfer.yaml" --ModelCls BertRCV1toNGDEVISE --data.args.setup_glove_embeddings true --model.args.checkpoint $CKPT_FOLDER/gile --data.args.val_label_json $CLSNAME "$@"
echo "SemLab"
$RUN_CMD --config "$CONFIG_FOLDER/ng_transfer.yaml" --model.args.checkpoint $CKPT_FOLDER/semlab_descriptions_n_10 --data.args.val_label_json $TRAIN "$@"
echo "Descr1"
$RUN_CMD --config "$CONFIG_FOLDER/ng_transfer.yaml" --model.args.checkpoint $CKPT_FOLDER/semlab_descriptions_n_1 --data.args.val_label_json $TRAIN "$@"
echo "classname"
$RUN_CMD --config "$CONFIG_FOLDER/ng_transfer.yaml" --model.args.checkpoint $CKPT_FOLDER/semlab_classnames --data.args.val_label_json $CLSNAME "$@"
