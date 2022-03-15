"""
Utils for multi-label text classification
"""

import itertools
from typing import NamedTuple, Union, Tuple
import numpy as np
import os
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)


def modify_config(config, data_args, model_args=None):
    if data_args.task_name == 'rcv1':
        config.problem_type = "multi_label_classification"

    # Add attributes from data_args to config
    # If they already exist in the config, then use that
    attributes_to_add = ['freeze_label_model', 'share_label_model', 'relative_weight_positive_samples', 'label_model_hidden_size', 'normalize_label_embeddings', \
                        'use_hinge_loss', 'hinge_margin_value', 'label_loss_type', 'focal_loss_gamma', 'use_gile']

    for attribute in attributes_to_add:
        if attribute not in dir(config):
            setattr(config, attribute, getattr(data_args, attribute))

    if not config.label_model_hidden_size and not data_args.share_label_model and model_args:
        label_config = AutoConfig.from_pretrained(data_args.label_model_name_or_path, cache_dir=model_args.cache_dir)
        config.label_model_hidden_size = label_config.hidden_size

    return config


def get_label_lists(datasets):
    """
    Get the list of labels being used in train, validation, and test datasets
    """

    label_list_dict = dict()

    for key in datasets:
        label_list_dict[key] = list(set(itertools.chain(*datasets[key]["bip:topics:1.0"])))
        label_list_dict[key].sort()

    return label_list_dict


def get_label_embedding_model(model, data_args, model_args):
    """
    Get the config, tokenizer, and model for label embedding
    """

    # Instantiate the tokenizer
    label_tokenizer = AutoTokenizer.from_pretrained(
        data_args.label_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.label_tokenizer = label_tokenizer

    # Print a warning about where the label model is coming from
    print("*********** Source of label model ***********")
    if data_args.use_label_model_from_checkpoint:
        print("Being initialized from checkpoint: {}".format(model_args.model_name_or_path))
    else:
        print("Being initialized from data_args.label_model_name_or_path: {}".format(data_args.label_model_name_or_path))

        label_model = AutoModel.from_pretrained(
            data_args.label_model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        model.label_model = label_model        
    print("*********************************************")

    return model


class EvalPredictionLabelDescriptions(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
        represented_labels: Labels chosen among all the global labels.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    represented_labels: np.ndarray