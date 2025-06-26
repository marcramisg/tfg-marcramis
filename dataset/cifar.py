from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import Compose

from .dataloader_generator import BaseDataLoaderGenerator


class Cifar10DataLoaderGenerator(BaseDataLoaderGenerator):
    """Data loader generator for the CIFAR10 dataset."""

    def __init__(
        self,
        data_dir: str | Path = "royalflush_datasets/cifar10",
        batch_size: int = 32,
        train_size: float = 0.8,
        transform: None | Compose = None,
    ) -> None:
        """Initializes the CIFAR10 data loader generator.

        Args:
            data_dir (str | Path): Directory where CIFAR10 data will be stored.
            batch_size (int): Batch size for data loaders.
            train_size (float): Proportion of data to use for training.
            transform (None | Compose): Transformations to apply to the data.
        """
        super().__init__(
            dataset_cls=datasets.CIFAR10,
            data_dir=data_dir,
            batch_size=batch_size,
            train_size=train_size,
            transform=transform,
        )


class Cifar100DataLoaderGenerator(BaseDataLoaderGenerator):
    """Data loader generator for the CIFAR100 dataset."""

    def __init__(
        self,
        data_dir: str | Path = "royalflush_datasets/cifar100",
        batch_size: int = 32,
        train_size: float = 0.8,
        transform: None | Compose = None,
    ) -> None:
        """Initializes the CIFAR100 data loader generator.

        Args:
            data_dir (str | Path): Directory where CIFAR100 data will be stored.
            batch_size (int): Batch size for data loaders.
            train_size (float): Proportion of data to use for training.
            transform (None | Compose): Transformations to apply to the data.
        """
        super().__init__(
            dataset_cls=datasets.CIFAR100,
            data_dir=data_dir,
            batch_size=batch_size,
            train_size=train_size,
            transform=transform,
        )


class Cifar8DataLoaderGenerator(BaseDataLoaderGenerator):
    """Data loader generator for the CIFAR8 dataset."""

    def __init__(
        self,
        data_dir: str | Path = "royalflush_datasets/cifar8",
        batch_size: int = 32,
        train_size: float = 0.8,
        transform: None | Compose = None,
    ) -> None:
        """Initializes the CIFAR100 data loader generator.

        Args:
            data_dir (str | Path): Directory where CIFAR100 data will be stored.
            batch_size (int): Batch size for data loaders.
            train_size (float): Proportion of data to use for training.
            transform (None | Compose): Transformations to apply to the data.
        """
        super().__init__(
            dataset_cls=Cifar8,
            data_dir=data_dir,
            batch_size=batch_size,
            train_size=train_size,
            transform=transform,
        )


class CifarN(datasets.CIFAR100):  # NOTE: It is a Dataset! Not DataLoaderGenerator
    def __init__(
        self,
        root,
        selected_classes_names: list[str],
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        if len(selected_classes_names) == 0:
            raise ValueError("selected_classes_names must have content.")

        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.targets: list[int]
        self.class_to_idx: dict[str, int]

        selected_classes = {
            k: v for (k, v) in self.class_to_idx.items() if k in selected_classes_names
        }  # example: {'bicycle': 8, 'dolphin': 30, 'motorcycle': 48, 'ray': 67, 'shark': 73, 'tank': 85, 'tractor': 89, 'trout': 91}
        self.class_to_idx = {
            k: selected_classes_names.index(k) for (k, v) in self.class_to_idx.items() if k in selected_classes_names
        }  # example: {'bicycle': 4, 'dolphin': 3, 'motorcycle': 5, 'ray': 0, 'shark': 2, 'tank': 6, 'tractor': 7, 'trout': 1}
        self.original_class_mapping = {
            selected_classes[k]: self.class_to_idx[k] for k in selected_classes.keys()
        }  # example: {8: 4, 30: 3, 48: 5 ...}

        # Filter and remap classes
        mask = [target in self.original_class_mapping for target in self.targets]
        self.data: np.ndarray = self.data[mask]  # ndarray[N(num-samples), 32, 32, 3]
        self.targets = [
            self.original_class_mapping[target] for target in self.targets if target in self.original_class_mapping
        ]


class Cifar8(CifarN):  # NOTE: It is a Dataset! Not DataLoaderGenerator
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.selected_classes_names = [
            "ray",
            "trout",
            "shark",
            "dolphin",
            "bicycle",
            "motorcycle",
            "tank",
            "tractor",
        ]

        super().__init__(
            root,
            selected_classes_names=self.selected_classes_names,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def get_subset(self, labels: Sequence[int | str]) -> Subset:
        """
        Returns a torch.utils.data.Subset containing only the filtered
        data that match the labels passed by the argument.

        Args:
            labels (list[str] | list[int]): list of label names or label IDs.

        Returns:
            Subset: Torch Subset containing the filtered data.
        """
        idx_labels: list[int] = [self.class_to_idx[lbl] if isinstance(lbl, str) else lbl for lbl in labels]

        filtered_indices_by_label = [idx for idx, lbl in enumerate(self.targets) if lbl in idx_labels]
        return Subset(self, filtered_indices_by_label)

    @staticmethod
    def get_labels_of_superclasses() -> dict[str, list[int]]:
        animal_indices = [0, 1, 2, 3]
        vehicle_indices = [4, 5, 6, 7]
        return {"animals": animal_indices, "vehicles": vehicle_indices}
