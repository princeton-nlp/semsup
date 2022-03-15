"""
Modified Newsgroups dataset where hold out certain classes for validation and testing
"""
from dataclasses import dataclass
from datasets import load_from_disk, DatasetDict

from .base import NewsgroupsDataArgs, NewsgroupsDataModule


@dataclass
class NewsgroupsHeldoutArgs(NewsgroupsDataArgs):
    """Data arguments for Newsgroups ZSLDM.
    Args:
        val_names (tuple): val classes to holdout from training
        test_names (tuple): test classes to hold out from training
        gzsl (bool): if set to true, val and test classes will be self.classes, otherwise it will be self.val/test_classes
        eval_train (bool): if set to true, eval on train classes
    """

    val_names: tuple = (
        "alt.atheism",
        "comp.sys.mac.hardware",
        "rec.motorcycles",
        "sci.electronics",
    )
    test_names: tuple = (
        "comp.os.ms-windows.misc",
        "rec.sport.hockey",
        "sci.space",
        "talk.politics.guns",
    )
    gzsl: bool = False
    eval_train: bool = False

    # filled in for you in __post_init__
    heldout_classes: tuple = None

    def __post_init__(self):
        super().__post_init__()
        self.heldout_classes = tuple(list(self.val_names) + list(self.test_names))
        self.train_classes = tuple(
            [x for x in self.classes if x not in self.heldout_classes]
        )
        self.val_classes = (
            tuple([x for x in self.classes if x not in self.test_names])
            if self.gzsl
            else self.val_names
        )
        self.test_classes = (
            tuple()
            if self.gzsl
            else self.test_names
        )

        if self.run_test:
            self.val_names = self.test_names
            self.val_classes = self.test_classes

        if self.eval_train:
            self.val_names = self.train_classes
            self.val_classes = self.train_classes


class NewsgroupsHeldoutDM(NewsgroupsDataModule):
    def __init__(self, args: NewsgroupsHeldoutArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)

    def setup(self, stage=None):
        self.setup_labels()

        # input dataset
        dataset = DatasetDict()
        loaded_dataset = load_from_disk(self.args.dataset_cache_path)
        dataset["train"] = loaded_dataset["train"].filter(
            lambda x: x["labels"] in self.train_classlabel.names
        )
        dataset["train"] = dataset["train"].map(
            lambda x: {"labels": self.train_classlabel.str2int(x["labels"])}
        )

        # we can pull from the larger train set, because the val/test classes are different
        split = "test" if self.args.run_test and self.args.eval_train else "train"
        
        dataset["val"] = loaded_dataset[split].filter(
            lambda x: x["labels"] in self.args.val_names
        ) 
        dataset["val"] = dataset["val"].map(
            lambda x: {"labels": self.val_classlabel.str2int(x["labels"])}
        )
        dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        )
        self.dataset = dataset
