#!/bin/bash

UNIT_TEST="--trainer.fast_dev_run true --trainer.progress_bar_refresh_rate 0 --trainer.checkpoint_callback false --trainer.logger false"

echo "SEMSUP"
bash run_cifar_semsup.sh $UNIT_TEST
echo "SUP"
bash run_cifar_sup.sh $UNIT_TEST
echo "CLSNAME"
bash run_cifar_clsname.sh $UNIT_TEST
echo "DESCR1"
bash run_cifar_descr1.sh $UNIT_TEST
echo "DEVISE"
bash run_cifar_devise.sh $UNIT_TEST
echo "GILE"
bash run_cifar_gile.sh $UNIT_TEST