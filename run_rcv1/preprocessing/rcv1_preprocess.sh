# Collate all files
python convert_raw_rcv1_to_json.py --path_to_rcv1 ../rcv1 --save_name rcv1_all.json --save_dir ../rcv1

# Split into train/val/test
python create_train_val_test_splits.py --train_fraction 0.60 --val_fraction 0.20 --seed 42 --file ../rcv1/rcv1_all.json

# Files for scenario 2
python create_rcv1_heldout_split_stratified.py --train_fraction 0.75 --seed 42 --file ../rcv1/rcv1_all.json --save_dir ../rcv1

# Files for scenario 3
python create_rcv1_superclass_split.py --seed 42 --file ../rcv1/rcv1_all.json