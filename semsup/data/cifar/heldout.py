""" zero-shot learning CIFAR100 datamodule
"""
from dataclasses import dataclass

from torch.utils.data import Subset
import torchvision
from torchvision.datasets.vision import StandardTransform
from sklearn.model_selection import train_test_split

from .base import CIFARDataArgs, CIFAR100DataModule


@dataclass
class CIFARHeldoutDataArgs(CIFARDataArgs):
    """Data arguments for CIFAR ZSL.
    Args:
        val_names (tuple): validation classes
        test_names (tuple): test classes
        gzsl (bool): if set to true, val classes will be self.classes, otherwise it will be self.heldout_classes
        eval_train (bool): if set to true, run eval on train classes
    """
    val_names: tuple = (
        "streetcar",
        "lamp",
        "forest",
        "otter",
        "house",
        "crab",
        "crocodile",
        "orchid",
        "rabbit",
        "man",
    )
    test_names: tuple = (
        "motorcycle",
        "pine_tree",
        "bottle",
        "trout",
        "chair",
        "butterfly",
        "chimpanzee",
        "orange",
        "leopard",
        "possum",
    )
    gzsl: bool = False  # generalized ZSL setting
    eval_train: bool = False

    # filled in for you in __post_init__()
    heldout_classes: tuple = None # val_names and class_names union

    def __post_init__(self):
        super().__post_init__()

        self.heldout_classes = tuple(list(self.val_names) + list(self.test_names))
        for n in self.heldout_classes:  # sanity check
            assert n in self.classes, f"{n} not a valid class name"

        self.train_classes = tuple(
            [x for x in self.classes if x not in self.heldout_classes]
        )
        self.val_classes = self.classes if self.gzsl else self.val_names

        if self.run_test:
            self.val_classes = self.test_names
            self.val_names = self.test_names

        if self.eval_train:
            self.val_classes = self.train_classes
            self.val_names = self.train_classes


class CIFARHeldoutDM(CIFAR100DataModule):
    """class which iplements zero-shot cifar"""

    def __init__(self, args: CIFARHeldoutDataArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)

    def setup(self, stage=None):
        self.setup_labels()

        # get the dataset
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir, train=True, download=False
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir, train=True, download=False
        )

        # class_ids that should not be in training / val
        train_heldout_ids = [
            self.all_classlabel.str2int(x) for x in self.args.heldout_classes
        ]
        val_heldout_ids = [
            self.all_classlabel.str2int(x) for x in self.args.classes if x not in self.args.val_names
        ]

        # filter the train and val examples
        train_idx_heldout, val_idx_heldout = [], []
        for idx in range(len(train_dataset)):
            _, class_id = train_dataset[idx]
            if class_id in train_heldout_ids:
                continue
            train_idx_heldout.append(idx)

        for idx in range(len(val_dataset)):
            _, class_id = val_dataset[idx]
            if class_id in val_heldout_ids:
                continue
            val_idx_heldout.append(idx)

        train_dataset = Subset(train_dataset, train_idx_heldout)
        val_dataset = Subset(val_dataset, val_idx_heldout)

        # get the label transforms for ZSL
        def train_target_transform(class_id: int) -> int:
            # converts the id from the original class label to the correct zsl training set label
            return self.train_classlabel.str2int(self.all_classlabel.int2str(class_id))

        def val_target_transform(class_id: int) -> int:
            return self.val_classlabel.str2int(self.all_classlabel.int2str(class_id))

        # add transforms to the datasets
        # we can't do this earlier, because we need the original target_transforms to do the
        # heldout sets
        val_dataset.dataset.transform = self.eval_transform
        val_dataset.dataset.target_transform = val_target_transform
        val_dataset.dataset.transforms = StandardTransform(
            transform=self.eval_transform, target_transform=val_target_transform
        )
        train_dataset.dataset.transform = self.train_transform
        train_dataset.dataset.target_transform = train_target_transform
        train_dataset.dataset.transforms = StandardTransform(
            transform=self.train_transform, target_transform=train_target_transform
        )
        self.dataset["train"] = train_dataset
        self.dataset["val"] = val_dataset

        if self.args.run_test and self.args.eval_train:
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.args.cache_dir, train=False, download=False
            )
            test_heldout_ids = [
                self.all_classlabel.str2int(x) for x in self.args.classes if x not in self.args.val_names
            ]
            test_idx_heldout = []
            for idx in range(len(test_dataset)):
                _, class_id = test_dataset[idx]
                if class_id in test_heldout_ids:
                    continue
                test_idx_heldout.append(idx)

            test_dataset = Subset(test_dataset, test_idx_heldout)
            test_dataset.dataset.transform = self.eval_transform
            test_dataset.dataset.target_transform = val_target_transform
            test_dataset.dataset.transforms = StandardTransform(
                transform=self.eval_transform, target_transform=val_target_transform
            )
            self.dataset["val"] = test_dataset


if __name__ == "__main__":
    # unit tests
    data_args = CIFARHeldoutDataArgs(
        label_tokenizer="prajjwal1/bert-small",
        train_label_json="../class_descrs/cifar/google_cifar100_autoclean.labels",
        cache_dir="../data_cache",
    )
    data_mod = CIFARHeldoutDM(args=data_args)
    data_mod.prepare_data()
    data_mod.setup()
    for _ in data_mod.train_dataloader():
        pass
    for _ in data_mod.val_dataloader():
        pass
