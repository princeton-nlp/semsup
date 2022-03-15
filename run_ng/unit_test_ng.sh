#!/bin/bash

UNIT_TEST="--trainer.fast_dev_run true --trainer.progress_bar_refresh_rate 0 --trainer.checkpoint_callback false --trainer.logger false"

echo "SEMSUP"
bash run_ng_semsup.sh $UNIT_TEST
echo "SUP"
bash run_ng_sup.sh $UNIT_TEST
echo "CLSNAME"
bash run_ng_clsname.sh $UNIT_TEST
echo "DESCR1"
bash run_ng_descr1.sh $UNIT_TEST
echo "DEVISE"
bash run_ng_devise.sh $UNIT_TEST
echo "GILE"
bash run_ng_gile.sh $UNIT_TEST