""" base CIFAR100 datamodule
"""
from dataclasses import dataclass
from pathlib import Path

from datasets import ClassLabel
import torchvision
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

from .base import AWADataset
from .defaults import AWA_TRAIN_CLASSES, AWA_VAL_CLASSES, AWA_TEST_CLASSES, AWA_CIFAR_CLASSES
from ..core import SemSupDataArgs, SemSupDataModule


@dataclass
class AWAHeldoutArgs(SemSupDataArgs):
    """Base argument dataclass for all AWA tasks
    Args:
        data_url (str): url from which to pull AWA data
        eval_train (bool): eval on the train classes
        remove_cifar (bool): remove class overlap with cifar from val_classes
    """
    data_url: str = "http://cvml.ist.ac.at/AwA2/AwA2-data.zip"
    eval_train: bool = False
    remove_cifar: bool = False

    # filled in for you in __post_init__()
    dataset_dir: str = None

    def __post_init__(self):
        super().__post_init__()
        self.dataset_dir = str(
            Path(self.cache_dir).joinpath("Animals_with_Attributes2")
        )
        self.train_classes = AWA_TRAIN_CLASSES
        self.val_classes = AWA_VAL_CLASSES

        if self.run_test:
            self.val_classes = AWA_TEST_CLASSES
        if self.eval_train:
            self.val_classes = AWA_TRAIN_CLASSES
        if self.remove_cifar:
            self.val_classes = tuple([
                animal for animal in self.val_classes if animal not in AWA_CIFAR_CLASSES
            ])


class AWAHeldoutDM(SemSupDataModule):
    """Heldout variant on the AWA dataset"""

    def __init__(self, args: AWAHeldoutArgs, *margs, **kwargs):
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
        
        self.dataset["train"] = AWADataset(
            classlabel = self.train_classlabel,
            transform = self.train_transform,
            root = imgs_root,
        )
        self.dataset["val"] = AWADataset(
            classlabel = self.val_classlabel,
            transform = self.eval_transform,
            root = imgs_root,
        )
