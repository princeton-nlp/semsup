""" zero-shot learning CIFAR100 datamodule
"""
from dataclasses import dataclass, field

from torch.utils.data import Subset
import torchvision
from datasets import ClassLabel
from sklearn.model_selection import train_test_split

from .base import CIFARDataArgs, CIFAR100DataModule


@dataclass
class CIFARSuperClassDataArgs(CIFARDataArgs):
    """Arguments for the fine to coarse and coarse to fine tasks
    Args:
        train_level (str): either 'fine' or 'coarse'. Sets the level of the train classes
        val_level (str): see train_level
        superclasses (tuple): the superclasses
        superclasses_to_classes (dict): the superclass to class mapping
    """

    train_level: str = "fine"
    val_level: str = "coarse"
    superclasses: tuple = (
        "aquatic_mammals",
        "fish",
        "flowers",
        "food_containers",
        "fruit_and_vegetables",
        "household_electrical_devices",
        "household_furniture",
        "insects",
        "large_carnivores",
        "large_man-made_outdoor_things",
        "large_natural_outdoor_scenes",
        "large_omnivores_and_herbivores",
        "medium_mammals",
        "non-insect_invertebrates",
        "people",
        "reptiles",
        "small_mammals",
        "trees",
        "vehicles_1",
        "vehicles_2",
    )
    val_superclasses: tuple = (
        "large_omnivores_and_herbivores",
        "people",
        "medium_mammals",
        "large_man-made_outdoor_things",
        "insects",
        "household_electrical_devices",
        "food_containers",
        "fish",
        "flowers",
        "vehicles_2",
    )
    test_superclasses: tuple = (
        "small_mammals",
        "reptiles",
        "non-insect_invertebrates",
        "large_natural_outdoor_scenes",
        "large_carnivores",
        "household_furniture",
        "fruit_and_vegetables",
        "aquatic_mammals",
        "trees",
        "vehicles_1",
    )
    superclass_to_classes: dict = field(
        default_factory=lambda: {
            "aquatic_mammals": ["otter", "beaver", "whale", "dolphin", "seal"],
            "fish": ["trout", "aquarium_fish", "shark", "flatfish", "ray"],
            "flowers": ["poppy", "rose", "orchid", "sunflower", "tulip"],
            "food_containers": ["plate", "bowl", "bottle", "can", "cup"],
            "fruit_and_vegetables": [
                "orange",
                "apple",
                "pear",
                "sweet_pepper",
                "mushroom",
            ],
            "household_electrical_devices": [
                "clock",
                "keyboard",
                "telephone",
                "television",
                "lamp",
            ],
            "household_furniture": ["table", "chair", "couch", "wardrobe", "bed"],
            "insects": ["caterpillar", "bee", "cockroach", "beetle", "butterfly"],
            "large_carnivores": ["leopard", "lion", "tiger", "bear", "wolf"],
            "large_man-made_outdoor_things": [
                "house",
                "bridge",
                "skyscraper",
                "road",
                "castle",
            ],
            "large_natural_outdoor_scenes": [
                "forest",
                "cloud",
                "plain",
                "mountain",
                "sea",
            ],
            "large_omnivores_and_herbivores": [
                "kangaroo",
                "cattle",
                "elephant",
                "camel",
                "chimpanzee",
            ],
            "medium_mammals": ["raccoon", "fox", "porcupine", "possum", "skunk"],
            "non-insect_invertebrates": ["snail", "lobster", "spider", "worm", "crab"],
            "people": ["girl", "woman", "man", "baby", "boy"],
            "reptiles": ["turtle", "snake", "lizard", "crocodile", "dinosaur"],
            "small_mammals": ["mouse", "shrew", "hamster", "squirrel", "rabbit"],
            "trees": [
                "palm_tree",
                "willow_tree",
                "pine_tree",
                "oak_tree",
                "maple_tree",
            ],
            "vehicles_1": ["bus", "bicycle", "motorcycle", "train", "pickup_truck"],
            "vehicles_2": ["streetcar", "tank", "lawn_mower", "tractor", "rocket"],
        }
    )

    # filled in for you in __post_init__()
    classes_to_superclass: dict = None

    def __post_init__(self):
        super().__post_init__()
        assert set(self.superclasses) == set(
            list(self.val_superclasses) + list(self.test_superclasses)
        )
        assert self.train_level.lower() in ("fine", "coarse")
        assert self.val_level.lower() in ("fine", "coarse")

        self.classes_to_superclass = {  # map class -> superclass
            x: k for k, v in self.superclass_to_classes.items() for x in v
        }
        self.train_classes = (
            self.superclasses if self.train_level == "coarse" else self.classes
        )
        self.val_classes = (
            self.val_superclasses if self.val_level == "coarse" else self.classes
        )

        if self.run_test:
            self.val_classes = self.test_superclasses
            self.val_superclasses = self.test_superclasses


class CIFARSuperClassDM(CIFAR100DataModule):
    """class which iplements superclass cifar"""

    def __init__(self, args: CIFARSuperClassDataArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)

    def setup(self, stage=None):
        self.setup_labels()

        dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir,
            train=True,
            download=False,
        )  # download without any transforms to do train-val split

        # make stratified split of train and val datasets
        train_idx, val_idx = train_test_split(
            list(range(len(dataset))),
            test_size=self.args.val_size,
            random_state=self.args.split_seed,
            shuffle=True,
            stratify=dataset.targets,
        )

        def train_target_transform(class_id: int) -> int:
            if self.args.train_level.lower() == "fine":
                return class_id
            classname = self.all_classlabel.int2str(class_id)
            superclass = self.args.classes_to_superclass[classname]
            return self.train_classlabel.str2int(superclass)

        def val_target_transform(class_id: int) -> int:
            if self.args.val_level.lower() == "fine":
                return class_id
            classname = self.all_classlabel.int2str(class_id)
            superclass = self.args.classes_to_superclass[classname]
            return self.val_classlabel.str2int(superclass)

        # further filter the val dataset to remove instances where superclass not in self.val_superclass
        if self.args.val_level.lower() == "coarse":
            val_idx_heldout = []
            val_heldout_ids = [
                self.all_classlabel.str2int(x)
                for x in self.args.classes
                if self.args.classes_to_superclass[x] not in self.args.val_superclasses
            ]

            for idx in val_idx:
                _, class_id = dataset[idx]
                if class_id in val_heldout_ids:
                    continue
                val_idx_heldout.append(idx)
            val_idx = val_idx_heldout

        # get the dataset
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir,
            train=True,
            download=False,
            transform=self.train_transform,
            target_transform=train_target_transform,
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir,
            train=True,
            download=False,
            transform=self.eval_transform,
            target_transform=val_target_transform,
        )

        self.dataset["train"] = Subset(train_dataset, train_idx)
        self.dataset["val"] = Subset(val_dataset, val_idx)

        if self.args.run_test:
            self.dataset["test"] = torchvision.datasets.CIFAR100(
                root=self.args.cache_dir,
                train=False,
                download=False,
            )

            test_idx_heldout = []
            for idx in range(len(self.dataset["test"])):
                _, class_id = self.dataset["test"][idx]
                if class_id in val_heldout_ids:
                    continue
                test_idx_heldout.append(idx)
            
            # get the dataset again, but this time with the right target transform
            self.dataset["test"] = torchvision.datasets.CIFAR100(
                root=self.args.cache_dir,
                train=False,
                download=False,
                transform=self.eval_transform,
                target_transform=val_target_transform,
            )

            self.dataset["val"] = Subset(self.dataset["test"], test_idx_heldout)


if __name__ == "__main__":
    # unit tests
    data_args = CIFARSuperClassDataArgs(
        label_tokenizer="prajjwal1/bert-small",
        train_label_json="../class_descrs/cifar/google_cifar100_autoclean.labels",
        val_label_json="../class_descrs/cifar/cifar100_superclass_eval_labels.labels",
        cache_dir="../data_cache",
    )
    data_mod = CIFARSuperClassDM(args=data_args)
    data_mod.prepare_data()
    data_mod.setup()
    for _ in data_mod.train_dataloader():
        pass
    for _ in data_mod.val_dataloader():
        pass
