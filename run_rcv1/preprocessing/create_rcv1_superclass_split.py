"""
Remove super classes from the train dataset and keep it only in the validation dataset.

Example command: python create_rcv1_heldout_split.py --train_fraction 0.75 --seed 42
"""

import argparse
import jsonlines
from collections import Counter
import numpy as np
import random
import copy
import os
import json


def create_dataset_split(args):
    random.seed(args.seed)
    # Read the JSON file containing one JSON per line and store the dict

    all_docs = []

    with jsonlines.open(args.file) as reader:
        for obj in reader:
            all_docs.append(obj)

    # Get a list of all the labels    
    label_statistics = Counter()
    for doc in all_docs:
        label_statistics.update(doc['bip:topics:1.0'])
    all_labels = list(label_statistics)

    # Ignore superclass labels during training
    super_class_labels = ['C15', 'C151', 'C17', 'C18', 'C31', 'C33', 'C41', 'E12', 'E13', 'E14', 'E21', 'E31', 'E41', 'E51', 'G15', 'M13', 'M14']
    train_labels = [label for label in all_labels if label not in super_class_labels]

    # Remove labels in train_labels from the train data
    new_docs = []
    for doc in all_docs:
        doc['bip:topics:1.0'] = [topic for topic in doc['bip:topics:1.0'] if topic in train_labels]
        if len(doc['bip:topics:1.0']) != 0:
            new_docs.append(doc)

    # Create a new file
    # Store list of dicts as a json
    save_name = 'rcv1_superclass_{}.json'.format(args.seed)
    args.save_dir = os.path.split(args.file)[0]
    f = open(os.path.join(args.save_dir, save_name), 'w', encoding="ISO-8859-1")

    for document in new_docs:
        f.write(str(json.dumps(document)) + '\n')

    f.close()


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--file", default='', type=str, help="")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for splitting classes.")

    args = parser.parse_args()

    create_dataset_split(args)

if __name__ == '__main__':
    main()