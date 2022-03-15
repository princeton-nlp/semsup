"""
Modified Newsgroups dataset where we classify superclasses (e.g. comp) instead of
finer classes (e.g. comp.graphics)
"""
from dataclasses import dataclass, field
from datasets import load_from_disk

from .base import NewsgroupsDataArgs, NewsgroupsDataModule


@dataclass
class NewsgroupsSuperClassArgs(NewsgroupsDataArgs):
    """ Arguments for the fine to coarse and coarse to fine tasks
    Args:
        train_level (str): either 'fine' or 'coarse'. Sets the level of the train classes
        val_level (str): see train_level
        val_superclasses (tuple): list of superclasses used for validation
        test_superclasses (tuple): list of superclasses used for test
        superclasses (tuple): the superclasses 
        superclasses_to_classes (dict): the superclass to class mapping
    """
    train_level: str = "fine"
    val_level: str = "coarse"
    val_superclasses: tuple = ("rec", "religion")
    test_superclasses: tuple = ("comp", "sci", "politics")
    superclass_to_classes: dict = field(
        default_factory=lambda: {
            "comp": [
                "comp.graphics",
                "comp.os.ms-windows.misc",
                "comp.sys.ibm.pc.hardware",
                "comp.sys.mac.hardware",
                "comp.windows.x"
            ],
            "rec": [
                "rec.autos",
                "rec.motorcycles",
                "rec.sport.baseball",
                "rec.sport.hockey",
            ],
            "sci": [
                "sci.crypt",
                "sci.electronics",
                "sci.med",
                "sci.space",
            ],
            "politics": [
                "talk.politics.guns",
                "talk.politics.mideast",
                "talk.politics.misc",
            ],
            "religion": [
                "soc.religion.christian",
                "talk.religion.misc",
                "alt.atheism",
            ],
            "misc": [
                "misc.forsale",
            ],
        }
    )

    # filled in for you in __post_init__()
    superclasses: tuple = None
    classes_to_superclass: dict = None

    def __post_init__(self):
        super().__post_init__()
        self.superclasses = tuple(list(self.val_superclasses) + list(self.test_superclasses))
        self.classes_to_superclass = {
            x: k for k, v in self.superclass_to_classes.items() for x in v
        }
        assert self.train_level.lower() in ("fine", "coarse")
        assert self.val_level.lower() in ("fine", "coarse")
        self.train_classes = (
            self.superclasses if self.train_level == "coarse" else self.classes
        )
        self.val_classes = (
            self.val_superclasses if self.val_level == "coarse" else self.classes
        )

        if self.run_test:
            self.val_classes = self.test_superclasses
            self.val_superclasses = self.test_superclasses


class NewsgroupsSuperClassDM(NewsgroupsDataModule):
    def __init__(self, args: NewsgroupsSuperClassArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)

    def setup(self, stage=None):
        self.setup_labels()

        dataset = load_from_disk(self.args.dataset_cache_path)

        if self.args.train_level == "coarse":
            dataset["train"] = dataset["train"].filter(
                lambda x: self.args.class_to_superclass[x["labels"]] in self.train_classlabel.names
            )
            dataset["train"] = dataset["train"].map(
                lambda x: {
                    "labels": self.train_classlabel.str2int(
                        self.args.classes_to_superclass[x["labels"]]
                    )
                }
            )
        else:
            dataset["train"] = dataset["train"].map(
                lambda x: {"labels": self.train_classlabel.str2int(x["labels"])}
            )

        if self.args.val_level == "coarse":
            split = "test" if self.args.run_test else "val"
            dataset["val"] = dataset[split].filter(
                lambda x: self.args.classes_to_superclass[x["labels"]] in self.val_classlabel.names
            )
            dataset["val"] = dataset["val"].map(
                lambda x: {
                    "labels": self.val_classlabel.str2int(
                        self.args.classes_to_superclass[x["labels"]]
                    )
                }
            )
        else:
            dataset["val"] = dataset["val"].map(
                lambda x: {"labels": self.val_classlabel.str2int(x["labels"])}
            )

        dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        )
        self.dataset = dataset
