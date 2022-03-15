"""
Dataloaders for loading the label descriptions to be passed through the label embedding model.
"""

from pathlib import Path
import json
import random
from collections import defaultdict, OrderedDict
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import gensim.downloader


class TokenLabelDataset(IterableDataset):
    '''
    Iterable dataset that infinitely interates over randomly sampled labels.
    returns tokenized outputs for use with fintetuning label model
    '''
    def __init__(self, class_labels, tokenized_labels, label_order_dict, key_to_use='input_ids'):
        super().__init__()
        
        # List of class labels
        self.class_labels = class_labels
        self.tokenized_labels = tokenized_labels

        # Create a mapping between order of label names to the integer id
        # For example, if the first class name in self.label_list_dict['train'] corresponds to integer ID 3, then store {1: 3}
        self.label_order_dict = torch.LongTensor([value for _, value in label_order_dict.items()])

        # Get number of label descriptions for each class
        self.num_descriptions = {
            c : tokenized_labels[c][key_to_use].shape[0] for c in self.class_labels
        }
        
    def __next__(self):
        fused = defaultdict(list)
        for c in self.class_labels:
            choice = random.choice(range(self.num_descriptions[c]))
            # Choose the right descriptions for all the keys like `input_ids`, `attention_mask` etc.
            for k, v in self.tokenized_labels[c].items():
                fused[k].append(v[choice]) # choose random label
        # each key will return a tensor (num_classes, max_seq_len)
        label_output = {k: torch.stack(v) for k, v in fused.items()}

        # Add a key to `label_output`
        label_output['represented_labels'] = self.label_order_dict
        
        return label_output

    def __iter__(self):
        return self


class LabelDescriptionsDataloaderBase:
    """
    Base dataloader for outputting label descriptions.
    Other variants should sub-class this.

    This dataloader uses the same descriptions for train, validation, and test.
    """

    def __init__(self, data_args, training_args, tokenizer, label_list_dict, label2id, id2label):
        """
        Args
        tokenizer: Tokenizer to use for label descriptions.
        label_list_dict: List of labels in train, validation, and test datasets.
        label2id: Label name to integer ID that is fixed globally. This is useful for evaluation.
        """
        # Store all the arguments
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.label_list_dict = label_list_dict
        self.label2id = label2id
        self.id2label = id2label
        self.num_workers = self.training_args.dataloader_num_workers

        # Splits
        self.splits = list()
        if training_args.do_train:
            self.splits.append('train')
        if training_args.do_eval:
            self.splits.append('validation')
        if training_args.do_predict:
            self.splits.append('test')

        # Create the train, val, and test datasets
        # self.datasets is a dictionary with keys `train`, `validation`, and `test`
        # self.label_order_dict is a dictionary containing mapping from label to the integer ID corresponding to that label
        self.datasets, self.label_order_dict = self.get_datasets()


    def get_train_dataset(self, split='train'):

        # Get the train dataset
        with Path(self.data_args.label_descriptions_train_file).open() as f:
            train_labels = json.load(f)
        train_labels.pop('Root', None)            

        # Tokenize the train dataset
        padding = "max_length" if self.data_args.pad_to_max_length else False
        max_seq_length = min(self.data_args.label_max_seq_length, self.tokenizer.model_max_length)
       
        tokenized_labels = {
            c : self.tokenizer(
                train_labels[c], truncation=True, padding=padding, max_length=max_seq_length, return_tensors='pt')
            for c in self.label_list_dict['train']
        }
        # Create a mapping between order of label names to the integer id
        # For example, if the first class name in self.label_list_dict['train'] corresponds to integer ID 3, then store {1: 3}
        label_order_dict = OrderedDict([(idx, self.label2id[value]) for idx, value in enumerate(self.label_list_dict['train'])])
        
        # Create the dataset for label descriptions
        label_dataset = TokenLabelDataset(class_labels=self.label_list_dict['train'], tokenized_labels=tokenized_labels, label_order_dict=label_order_dict)

        return label_dataset, label_order_dict


    def get_validation_dataset(self, split='validation'):

        # Get the validation label dataset
        label_file_name = getattr(self.data_args, 'label_descriptions_{}_file'.format(split))
        label_file = label_file_name if label_file_name else self.data_args.label_descriptions_train_file

        with Path(label_file).open() as f:
            validation_labels = json.load(f)
        validation_labels.pop('Root', None)

        # Tokenize the train dataset
        padding = "max_length" if self.data_args.pad_to_max_length else False
        max_seq_length = min(self.data_args.label_max_seq_length, self.tokenizer.model_max_length)

        # Create a mapping between order of label names to the integer id
        # For example, if the first class name in self.label_list_dict[split] corresponds to integer ID 3, then store {1: 3}
        # Check if it is the zero-shot setting or the generalized zero-shot setting
        if self.data_args.evaluation_type == 'zs':
            label_order_dict = OrderedDict([(idx, self.label2id[value]) for idx, value in enumerate(self.label_list_dict[split])])
            class_labels = self.label_list_dict[split]

            tokenized_labels = {
                c : self.tokenizer(
                    validation_labels[c], truncation=True, padding=padding, max_length=max_seq_length, return_tensors='pt')
                for c in self.label_list_dict[split]
            }
        elif self.data_args.evaluation_type == 'gzs':
            # Make sure all the classes are represented in the label descriptions
            assert len(validation_labels) == len(self.label2id), "Some classes are missing in the label descriptions file of {}.\
                                                                Only following classes represented: {} \
                                                                Total classes = {}. Expected = {}".format(split, validation_labels.keys(), len(validation_labels), len(self.label2id))
            label_order_dict = OrderedDict([(idx, idx) for idx in range(len(self.label2id))])
            class_labels = [self.id2label[idx] for idx in range(len(self.id2label))]

            tokenized_labels = {
                self.id2label[c] : self.tokenizer(
                    validation_labels[self.id2label[c]], truncation=True, padding=padding, max_length=max_seq_length, return_tensors='pt')
                for c in label_order_dict.keys()
            }         
        
        # Create the dataset for label descriptions
        label_dataset = TokenLabelDataset(class_labels=class_labels, tokenized_labels=tokenized_labels, label_order_dict=label_order_dict)

        return label_dataset, label_order_dict


    def get_test_dataset(self, split='test'):
        return self.get_validation_dataset(split)


    def get_datasets(self):
        # Open all the datasets specified
        # The train dataset is the dataset for all splits
        datasets = dict()
        label_order_dict = OrderedDict()

        # Iterate over the required functions
        # self.split is usually ['train', 'validation', 'test']
        for key in self.splits:
            function = getattr(self, 'get_{}_dataset'.format(key))
            dataset, label_order = function()
            datasets[key] = dataset
            label_order_dict[key] = label_order

        return datasets, label_order_dict
        

    def get_dataloader(self, split='train'):
        """
        Return the dataloader corresponding to the split name
        """
        return DataLoader(self.datasets[split], num_workers=self.num_workers)


class LabelDescriptionsDataloaderWord2Vec(LabelDescriptionsDataloaderBase):
    """
    Base dataloader for outputting word vectors for the label descriptions.
    Other variants should sub-class this.

    This dataloader uses the same descriptions for train, validation, and test.
    """

    def __init__(self, *args, **kwargs):
        """
        Args
        tokenizer: Tokenizer to use for label descriptions.
        label_list_dict: List of labels in train, validation, and test datasets.
        label2id: Label name to integer ID that is fixed globally. This is useful for evaluation.
        """

        # Load the word2vec model
        self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
        self.glove_vectors_keys = self.glove_vectors.key_to_index.keys()        

        super().__init__(*args, **kwargs)


    def get_average_embeddings(self, list_of_sentences):
        """
        Average word embeddings of words in a sentence and return a list.
        Only lowercase.

        Treating unknown words as zero vectors.
        """

        average_embeddings = []

        for sentence in list_of_sentences:
            sum_of_embeddings = 0.
            for word in sentence.strip().split():
                word = word.lower()
                if word in self.glove_vectors_keys:
                    sum_of_embeddings += self.glove_vectors.vectors[self.glove_vectors.get_index(word)]
            
            # Average the embeddings
            sum_of_embeddings /= len(sentence.strip().split())

            # Append to a list
            average_embeddings.append(sum_of_embeddings)

        average_embeddings = {'embeddings': torch.Tensor(average_embeddings)}

        return average_embeddings


    def get_train_dataset(self, split='train'):

        # Get the train dataset
        with Path(self.data_args.label_descriptions_train_file).open() as f:
            train_labels = json.load(f)
        train_labels.pop('Root', None)

        # Get the average embeddings for all the descriptions by using the gensim model
        average_embeddings = {c : self.get_average_embeddings(train_labels[c]) for c in self.label_list_dict['train']}

        # Create a mapping between order of label names to the integer id
        # For example, if the first class name in self.label_list_dict['train'] corresponds to integer ID 3, then store {1: 3}
        label_order_dict = OrderedDict([(idx, self.label2id[value]) for idx, value in enumerate(self.label_list_dict['train'])])
        
        # Create the dataset for label descriptions
        label_dataset = TokenLabelDataset(class_labels=self.label_list_dict['train'], tokenized_labels=average_embeddings, label_order_dict=label_order_dict, key_to_use='embeddings')

        return label_dataset, label_order_dict


    def get_validation_dataset(self, split='validation'):

        # Get the validation label dataset
        label_file_name = getattr(self.data_args, 'label_descriptions_{}_file'.format(split))
        label_file = label_file_name if label_file_name else self.data_args.label_descriptions_train_file

        with Path(label_file).open() as f:
            validation_labels = json.load(f)
        validation_labels.pop('Root', None)

        # Create a mapping between order of label names to the integer id
        # For example, if the first class name in self.label_list_dict[split] corresponds to integer ID 3, then store {1: 3}
        # Check if it is the zero-shot setting or the generalized zero-shot setting
        if self.data_args.evaluation_type == 'zs':
            label_order_dict = OrderedDict([(idx, self.label2id[value]) for idx, value in enumerate(self.label_list_dict[split])])
            class_labels = self.label_list_dict[split]

            average_embeddings = {c : self.get_average_embeddings(validation_labels[c]) for c in self.label_list_dict[split]}

        elif self.data_args.evaluation_type == 'gzs':
            # Make sure all the classes are represented in the label descriptions
            assert len(validation_labels) == len(self.label2id), "Some classes are missing in the label descriptions file of {}.\
                                                                Only following classes represented: {} \
                                                                Total classes = {}. Expected = {}".format(split, validation_labels.keys(), len(validation_labels), len(self.label2id))
            label_order_dict = OrderedDict([(idx, idx) for idx in range(len(self.label2id))])
            class_labels = [self.id2label[idx] for idx in range(len(self.id2label))]

            average_embeddings = {self.id2label[c] : self.get_average_embeddings(validation_labels[self.id2label[c]]) for c in label_order_dict.keys()}
        
        # Create the dataset for label descriptions
        label_dataset = TokenLabelDataset(class_labels=class_labels, tokenized_labels=average_embeddings, label_order_dict=label_order_dict, key_to_use='embeddings')

        return label_dataset, label_order_dict