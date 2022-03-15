"""
Run the DeVise word2vec baseline on RCV1

Script adapted from HuggingFace v4.14.1-release
run_glue.py: https://raw.githubusercontent.com/huggingface/transformers/v4.14.1-release/examples/pytorch/text-classification/run_glue.py

Example command: python -m pdb run_rcv1_word2vec_descriptions.py --model_name_or_path prajjwal1/bert-small --label_model_name_or_path prajjwal1/bert-small --cache_dir /n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models --max_seq_length 512 --do_train --do_eval --do_predict --use_label_descriptions --train_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_heldout/rcv1_train_75_42.json --validation_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_processed/rcv1_valid.json --test_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_processed/rcv1_test.json --label_descriptions_train_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/continuous-classification/labels/rcv1/google_rcv1_autoclean.json --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --learning_rate 2e-5 --lr_scheduler_type constant --max_steps 10 --output_dir /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/model_outputs/debug --overwrite_output_dir --save_steps -1 --evaluation_strategy steps --eval_steps 5000

Debug command: python -m pdb run_rcv1_word2vec_descriptions.py --model_name_or_path prajjwal1/bert-small --label_model_hidden_size 300 --cache_dir /n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models --max_seq_length 512 --do_train --do_eval --do_predict --use_label_descriptions --train_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_processed/rcv1_small_all_new.json --validation_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_processed/rcv1_small_all_new.json --test_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_processed/rcv1_small_all_new.json --label_descriptions_train_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/continuous-classification/labels/rcv1/google_rcv1_autoclean.json --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --learning_rate 2e-5 --lr_scheduler_type constant --max_steps 10 --output_dir /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/model_outputs/debug --overwrite_output_dir --save_steps -1

Only eval: python -m pdb run_rcv1_word2vec_descriptions.py --model_name_or_path /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/model_outputs/rcv1/s4_devise_stratified_split --use_label_model_from_checkpoint  --label_model_name_or_path prajjwal1/bert-small --cache_dir /n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models --max_seq_length 512 --do_eval --use_label_descriptions --train_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_heldout/all_labels_stratified.json --validation_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/data/RCV1/rcv1_processed/rcv1_small_all_new.json  --label_descriptions_train_file /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/continuous-classification/labels/rcv1/rcv1_only_class_name_no_template_bert.json --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --learning_rate 2e-5 --lr_scheduler_type constant --max_steps 10 --output_dir /n/fs/nlp-asd/asd/asd/Projects/SemanticLabelEmbeddings/model_outputs/debug --overwrite_output_dir --save_steps -1 --label_model_hidden_size 300
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import itertools

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Added imports
# Import the arguments
from src import (
    RCV1TrainerLabelDescriptions,
    DataTrainingArguments,
    ModelArguments,
    modify_config,
    multilabel_metrics,
    multilabel_label_descriptions_metrics,
    multilabel_label_descriptions_per_class_threshold_metrics,
    multilabel_label_descriptions_ranking_metrics,
    get_label_lists,
    LabelDescriptionsDataloaderBase,
    LabelDescriptionsDataloaderWord2Vec,
    get_label_embedding_model,
    AutoModelForSemanticEmbedding,
    AutoModelForWord2vec
)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.14.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)


    # Load the RCV1 datasets

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name == 'rcv1':
        # Use the data file which contains all the data so that the label2id and id2label dictionaries can be frozen
        all_data_dataset = load_dataset("json", data_files={"train": data_args.all_labels_file}, cache_dir=model_args.cache_dir)
        label_list = list(set(itertools.chain(*all_data_dataset["train"]["bip:topics:1.0"])))
    else:
        raise("Support available only for 'rcv1' dataset right now.")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    # TODO: Might have to get 'bip:industries:1.0' and 'bip:countries:1.0' labels as well

    # Get label lists for train, validation, and test
    label_list_dict = get_label_lists(raw_datasets)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Modify the config to mention that it is a multi-label classification problem
    config = modify_config(config, data_args, model_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForWord2vec.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the raw_datasets
    # RCV1 specific preprocessing
    if data_args.task_name == 'rcv1':
        sentence1_key, sentence2_key = "text", None
        label_key = "bip:topics:1.0"

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    label_to_id = model.config.label2id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
    
    # label_to_id = None
    # if (
    #     model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    #     and data_args.task_name is not None
    # ):
    #     # Some have all caps in their config, some don't.
    #     # BUG: This is a HF bug. Don't do lower here.
    #     # BUG: label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    #     label_name_to_id = {k: v for k, v in model.config.label2id.items()}
    #     if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
    #         label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
    #             "\nIgnoring the model labels as a result.",
    #         )
    # else:
    #     label_to_id = {v: i for i, v in enumerate(label_list)}

    # if label_to_id is not None:
    #     model.config.label2id = label_to_id
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}
    # elif data_args.task_name is not None:
    #     model.config.label2id = {l: i for i, l in enumerate(label_list)}
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}

    # Instantiate the label embedding model
    # model now has model.label_model_name, model.label_model, model.label_tokenizer
    # model = get_label_embedding_model(model, data_args, model_args)
    label_tokenizer = AutoTokenizer.from_pretrained(data_args.label_model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer)

    # Get the label dataset object
    label_descriptions_dataloader = LabelDescriptionsDataloaderWord2Vec(
        data_args,
        training_args,
        label_tokenizer,
        label_list_dict,
        model.config.label2id,
        model.config.id2label
    )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and label_key in examples:
            # NOTE: Multi-label dataset
            # 0 if that label is not present, 1 if the label is present.
            result["label"] = [[0 if model.config.id2label[i] not in l else 1 for i in range(len(label_to_id))] for l in examples[label_key]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict and (data_args.task_name is not None or data_args.test_file is not None):
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name == 'rcv1':
        compute_metrics = multilabel_label_descriptions_ranking_metrics(data_args, model.config.id2label, model.config.label2id, label_list_dict, {})
    else:
        raise("Only RCV1 supported.")

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = RCV1TrainerLabelDescriptions(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label_descriptions_dataloader=label_descriptions_dataloader,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            # Pop the key 'fbr' so that it can be passed to the predict block below
            fbr = metrics.pop('eval_fbr')

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [predict_dataset]

        # New metric function using the fbr computed on the validation set
        if data_args.task_name == 'rcv1':
            compute_metrics = multilabel_label_descriptions_ranking_metrics(data_args, model.config.id2label, model.config.label2id, label_list_dict, fbr)
        else:
            raise("Only RCV1 supported.")
        trainer.compute_metrics = compute_metrics

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="predict")
            metrics.pop("predict_fbr")

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()