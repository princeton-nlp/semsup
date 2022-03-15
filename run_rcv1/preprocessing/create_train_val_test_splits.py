"""
Create train, val, and test splits

Example command: python create_train_val_test_splits.py --train_fraction 0.60 --val_fraction 0.20 --seed 42 --file rcv1/rcv1_all.json
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

    # Randomly split into train, val, and test
    random.shuffle(all_docs)
    train_docs = all_docs[:int(len(all_docs) * args.train_fraction)]
    remaining = all_docs[int(len(all_docs) * args.train_fraction):]

    # Randomly split into val and test
    split = int(args.val_fraction / (1 - args.train_fraction) * len(remaining))
    val_docs = remaining[:split]
    test_docs = remaining[split:]

    # Store all the files
    files = [train_docs, val_docs, test_docs]

    for docs, name in zip(files, ['train', 'val', 'test']):
        save_name = 'rcv1_{}.json'.format(name)
        args.save_dir = os.path.split(args.file)[0]
        f = open(os.path.join(args.save_dir, save_name), 'w', encoding="ISO-8859-1")

        for document in docs:
            f.write(str(json.dumps(document)) + '\n')

        f.close()


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--file", default='/n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_processed/rcv1_train.json', type=str, help="")
    parser.add_argument("--train_fraction", default=0.75, type=float, help="Amount of data in the train dataset.")
    parser.add_argument("--val_fraction", default=0.20, type=float, help="Amount of data in the validation dataset.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for splitting classes.")

    args = parser.parse_args()

    create_dataset_split(args)

if __name__ == '__main__':
    main()