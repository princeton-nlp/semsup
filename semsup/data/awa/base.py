""" base CIFAR100 datamodule
"""
from dataclasses import dataclass
from pathlib import Path

from datasets import ClassLabel
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
from torchvision.datasets.utils import download_and_extract_archive
from sklearn.model_selection import train_test_split

from .defaults import AWA_ALL_CLASSES
from ..core import SemSupDataArgs, SemSupDataModule


def read_awa_file(file):
    """Read a names txt file from the AWA dataset"""
    all_names = []
    with Path(file).open() as f:
        for line in f:
            class_name = line.split("\t")[1].strip()
            all_names.append(class_name)
        return tuple(all_names)


class AWADataset(torchvision.datasets.ImageFolder):
    """ AWA Dataset constructed using ImageFolder.
    Args:
        classlabel (ClassLabel): The classlabel to define targets for the dataset. Classes not in the classlabel are removed.
    """
    def __init__(self, classlabel: ClassLabel, *args, **kwargs):
        self.classlabel = classlabel        
        super().__init__(*args, **kwargs)
        
    def find_classes(self, *args, **kwargs):
        super().find_classes(*args, **kwargs)
        classes = list(self.classlabel.names)
        cls_to_idx = dict()
        for cls in classes:
            cls_to_idx[cls] = self.classlabel.str2int(cls)
        return classes, cls_to_idx


@dataclass
class AWADataArgs(SemSupDataArgs):
    """Base argument dataclass for all AWA tasks
    Args:
        data_url (str): url from which to pull AWA data
        split_seed (int): seed to make train-val split
        val_size (float): size of the val set of each class (from train)
    """
    data_url: str = "http://cvml.ist.ac.at/AwA2/AwA2-data.zip"
    split_seed: int = 42  # seed to make train-val split
    val_size: float = 0.2  # size of the val set of each class (from train)

    # filled in for you in __post_init__()
    dataset_dir: str = None

    def __post_init__(self):
        super().__post_init__()
        self.dataset_dir = str(
            Path(self.cache_dir).joinpath("Animals_with_Attributes2")
        )
        self.train_classes = AWA_ALL_CLASSES
        self.val_classes = AWA_ALL_CLASSES
        self.split_seed:int = 42
        self.val_size: float = 0.2


class AWADataModule(SemSupDataModule):
    """Pytorch lightning datamodule"""

    def __init__(self, args: AWADataArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)

        #  stats
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # transforms
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def prepare_data(self):
        self.prepare_label_data()
        if not Path(self.args.dataset_dir).exists():
            download_and_extract_archive(
                url=self.args.data_url,
                download_root=self.args.cache_dir
            )

    def setup(self, stage=None):
        self.setup_labels()
        imgs_root = str(Path(self.args.dataset_dir).joinpath("JPEGImages"))
        
        train_dataset = AWADataset(
            classlabel = self.val_classlabel,
            transform = self.train_transform,
            root = imgs_root,
        )

        eval_dataset = AWADataset(
            classlabel = self.val_classlabel,
            transform = self.eval_transform,
            root = imgs_root,
        )

        train_idx, test_idx = train_test_split(
            list(range(len(train_dataset))),
            test_size=self.args.val_size,
            random_state=self.args.split_seed,
            shuffle=True,
            stratify=train_dataset.targets,
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=self.args.val_size,
            random_state=self.args.split_seed,
            shuffle=True,
            stratify=[train_dataset.targets[idx] for idx in train_idx],
        )

        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(set(test_idx).intersection(set(val_idx))) == 0

        self.dataset["train"] = Subset(train_dataset, train_idx)
        if self.args.run_test:
            self.dataset["val"] = Subset(eval_dataset, test_idx)
        else:
            self.dataset["val"] = Subset(eval_dataset, val_idx)


if __name__ == "__main__":
    # unit tests
    data_args = AWADataArgs(
        label_tokenizer="huggingface/CodeBERTa-small-v1",
        train_label_json="../class_descrs/awa/awa_base.labels",
        cache_dir="../data_cache",
    )
    data_mod = AWADataModule(args=data_args)
    data_mod.prepare_data()
    data_mod.setup()

    print("test_run dataloader")
    for _ in data_mod.train_dataloader():
        break
    for _ in data_mod.val_dataloader():
        break
