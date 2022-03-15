"""
Dataset code for the Newsgroups dataset
"""
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer
import datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk

from ..core import SemSupDataArgs, SemSupDataModule


@dataclass
class NewsgroupsDataArgs(SemSupDataArgs):
    """Base argument dataclass for all Newsgroups problems
    Args:
        input_tokenizer (str): name of the HF tokenizer to use to preprocess the dataset
        input_max_len (int): the maximum length used by the input tokenizerss
        train_classes (tuple): subset of classes. Will set to classes if None is specified
        val_classes (tuple): subset of classes. Will set to train classes if None is specified
        classes (tuple): names of all classes (must match name on Datasets Hub)
        split_seed (int): seed to make train-val split
        test_size (float): size of the test size of each class (from total)
        val_size (float): size of the val set of each class (from train)
        variant (str): variant of newsgroups to use
    """

    input_tokenizer: str = None
    input_max_len: int = 512
    train_classes: tuple = None
    val_classes: tuple = None
    classes: tuple = (
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    )
    split_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    variant: str = "18828"

    # these are filled in for you in fill_dataset_args()
    dataset_hash: str = None
    dataset_cache_path: str = None

    def __post_init__(self):
        super().__post_init__()
        if self.train_classes is None:
            self.train_classes = self.classes
        if self.val_classes is None:
            self.val_classes = self.train_classes

        self.dataset_hash = self.hash_func(
            (
                "newsgroups",
                sorted(list(self.classes)),
                self.input_max_len,
                self.split_seed,
                self.test_size,
                self.val_size,
                self.variant,
            )
        )
        self.dataset_cache_path = str(
            Path(self.cache_dir).joinpath("ng_" + self.dataset_hash)
        )


class NewsgroupsDataModule(SemSupDataModule):
    """Base Lightning DataModule for Newsgroups. Variants of Newsgroups data (such as superclasses
    or held out class zsl) should subclass this and override the setup() method to define new
    datasets and splits.

    args (NewsgroupsDataArgs): general arguments for the dataset
    batch_size (int): batch size used by Lightning Trainer
    val_batch_size (int or None): batch size for validation (if different from batch_size)
    num_workers (int): number of workers to spawn for the dataloader
    """

    def __init__(
        self,
        args: NewsgroupsDataArgs,
        batch_size: int = 64,
        val_batch_size: int = None,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(
            args=args,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            **kwargs,
        )

    def _configure_dataset(self, dataset, tokenizer, class_name: str):
        """Add labels to the newsgroups dataset and tokenize it"""
        dataset = dataset.map(
            lambda x: {"labels": len(x["text"]) * [class_name]}, batched=True
        )
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=self.args.input_max_len,
            ),
            batched=True,
        )
        return dataset

    def prepare_data(self):
        self.prepare_label_data()  # setups up label dataset
        if Path(self.args.dataset_cache_path).exists():
            return
        tokenizer = AutoTokenizer.from_pretrained(self.args.input_tokenizer)
        dataset_list = []
        for c in self.args.classes:
            dataset_list.append(
                self._configure_dataset(
                    load_dataset(
                        "newsgroup", f"{self.args.variant}_{c}", split="train"
                    ),
                    tokenizer,
                    class_name=c,
                )
            )
        combined = concatenate_datasets(dataset_list)
        dataset = datasets.DatasetDict()  # the dataset
        # make train-test split
        train_test = combined.train_test_split(
            test_size=self.args.test_size, seed=self.args.split_seed
        )
        dataset["test"] = train_test["test"]

        # split train set into train-val
        train_val = train_test["train"].train_test_split(
            test_size=self.args.val_size, seed=self.args.split_seed
        )
        dataset["train"] = train_val["train"]
        dataset["val"] = train_val["test"]
        dataset.save_to_disk(self.args.dataset_cache_path)

    def setup(self, stage=None):
        self.setup_labels()  # setup the label dataset

        # input dataset
        dataset = load_from_disk(self.args.dataset_cache_path)
        dataset = dataset.map(
            lambda x: {"labels": self.train_classlabel.str2int(x["labels"])}
        )
        dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        )
        if self.args.run_test:
            dataset["val"] = dataset["test"]
        self.dataset = dataset


if __name__ == "__main__":
    """Unit tests"""
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    data_args = NewsgroupsDataArgs(
        input_tokenizer="bert-base-uncased",
        label_tokenizer="prajjwal1/bert-small",
        train_label_json="../class_descrs/newsgroups/ng_base.labels",
        cache_dir="../data_cache",
    )
    data_mod = NewsgroupsDataModule(args=data_args)
    data_mod.prepare_data()
    data_mod.setup()
    for _ in data_mod.train_dataloader():
        pass
    for _ in data_mod.val_dataloader():
        pass
