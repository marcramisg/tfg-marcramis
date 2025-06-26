from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Compose

from ..datatypes.data import (
    DataLoaders,
    DatasetSettings,
    IidDatasetSettings,
    NonIidDirichletDatasetSettings,
    NonIidNonOverlappingClassesDatasetSettings,
)
from ..utils.random import RandomUtils


class DataloaderGeneratorInterface(object, metaclass=ABCMeta):

    @abstractmethod
    def get_dataloaders(self, settings: DatasetSettings) -> DataLoaders:
        """Gets the data loaders for IID or Non-IID data.

        Args:
            settings (DatasetSettings): The settings to generate the specified data loaders.

        Returns:
            DataLoaders: Dataclass containing the data loaders.
        """
        raise NotImplementedError

    def _build_dataloaders(self, batch_size: int, train: Dataset, validation: Dataset, test: Dataset) -> DataLoaders:
        """Builds data loaders for training, validation, and testing.

        Args:
            batch_size (int): Batch size for data loaders.
            train (Dataset): Training dataset.
            validation (Dataset): Validation dataset.
            test (Dataset): Test dataset.

        Returns:
            DataLoaders: Dataclass containing the data loaders.
        """
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
        return DataLoaders(
            train=train_loader,
            validation=validation_loader,
            test=test_loader,
        )


class BaseDataLoaderGenerator(DataloaderGeneratorInterface):
    """Base data loader generator class for handling IID and Non-IID data."""

    def __init__(
        self,
        dataset_cls: type,
        data_dir: str | Path,
        batch_size: int,
        train_size: float,
        transform: None | Compose = None,
    ) -> None:
        """Initializes the data loader generator.

        Args:
            dataset_cls (type): The dataset class (e.g., datasets.CIFAR10).
            data_dir (str | Path): Directory where data will be stored.
            batch_size (int): Batch size for data loaders.
            train_size (float): Proportion of data to use for training.
            transform (None | Compose): Transformations to apply to the data.
        """
        self.dataset_cls = dataset_cls
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.batch_size = batch_size
        self.train_size = train_size
        self.transform = transform or Compose([transforms.ToTensor()])

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.data_dir.resolve()

    def _get_datasets(self) -> Tuple[VisionDataset, VisionDataset]:
        """Loads the train and test datasets.

        Returns:
            Tuple[VisionDataset, VisionDataset]: Train and test datasets.
        """
        train_dataset = self.dataset_cls(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=True,
        )
        test_dataset = self.dataset_cls(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=True,
        )
        return train_dataset, test_dataset

    def _build_dataloaders_iid(self, settings: IidDatasetSettings) -> DataLoaders:
        """Builds IID data loaders.

        Returns:
            DataLoaders: Data loaders for IID data.
        """
        train_dataset: Subset
        test_dataset: Subset
        train_dataset, test_dataset = self._get_datasets()

        if not settings.are_all_train_samples_selected():
            train_dataset = settings.get_new_train_dataset(train_dataset)

        if not settings.are_all_test_samples_selected():
            test_dataset = settings.get_new_test_dataset(test_dataset)

        # Split train in train and validation
        total_train_samples = len(train_dataset)
        train_size = int(self.train_size * total_train_samples)
        validation_size = total_train_samples - train_size
        train_set, validation_set = random_split(train_dataset, [train_size, validation_size])

        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=test_dataset,
        )

    def _build_dataloaders_non_iid(self, settings: NonIidNonOverlappingClassesDatasetSettings) -> DataLoaders:
        """Builds Non-IID data loaders.

        Args:
            num_clients (int): Number of clients.
            classes_per_client (int): Number of classes per client.

        Returns:
            DataLoaders: Data loaders for Non-IID data.
        """
        train_dataset, test_dataset = self._get_datasets()
        num_classes = len(train_dataset.classes)
        print(f"num classes: {num_classes}")

        # Create a mapping from class indices to data indices
        targets = np.array(train_dataset.targets)
        data_indices = [np.where(targets == i)[0] for i in range(num_classes)]
        print(f"data_indices: {data_indices[:10]}")

        # Shuffle and distribute classes to clients
        classes = list(range(num_classes))
        np.random.shuffle(classes)
        if settings.num_clients * settings.classes_per_client <= num_classes:
            raise RuntimeError(
                f"Not enough classes {num_classes} for the number of clients "
                + f"{settings.num_clients} and {settings.classes_per_client}."
            )

        client_classes = [
            classes[i * settings.classes_per_client : (i + 1) * settings.classes_per_client]
            for i in range(settings.num_clients)
        ]

        # Collect indices for each client
        client_indices: list[int] = []
        for cls in client_classes:
            indices = np.hstack([data_indices[c] for c in cls])
            np.random.shuffle(indices)
            client_indices.extend(indices)

        # Split into training and validation sets
        all_train_indices = np.hstack(client_indices)
        train_size = int(self.train_size * len(all_train_indices))
        train_indices: list[int] = all_train_indices[:train_size].tolist()
        validation_indices: list[int] = all_train_indices[train_size:].tolist()

        train_set = Subset(train_dataset, train_indices)
        validation_set = Subset(train_dataset, validation_indices)

        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=test_dataset,
        )

    def _build_dataloaders_non_iid_dirichlet(self, settings: NonIidDirichletDatasetSettings) -> DataLoaders:
        """Builds Non-IID data loaders using a Dirichlet distribution.

        Args:
            num_clients (int): Number of clients.
            alpha (float): Parameter controlling the level of non-IID-ness.

        Returns:
            DataLoaders: Data loaders for Non-IID data.
        """
        num_clients = settings.num_clients
        client_index = settings.client_index

        train_dataset, test_dataset = self._get_datasets()
        num_classes = len(train_dataset.classes)
        targets = np.array(train_dataset.targets)
        # data_indices = np.arange(len(train_dataset))

        # Generate Dirichlet distribution
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
        client_data_indices: list[list] = [[] for _ in range(num_clients)]

        for _, indices in enumerate(class_indices):
            proportions = np.random.dirichlet(np.repeat(settings.dirichlet_alpha, num_clients))
            proportions = (proportions * len(indices)).astype(int)
            proportions[-1] = len(indices) - sum(proportions[:-1])  # Adjust last client
            split_indices = np.split(indices, np.cumsum(proportions)[:-1])

            # for i, client_indices_i in enumerate(split_indices):
            #     client_data_indices[i].extend(client_indices_i)
            client_data_indices[client_index].extend(split_indices[client_index])

        # Flatten and shuffle indices
        # all_train_indices = np.hstack(client_data_indices)
        np.random.shuffle(client_data_indices[client_index])
        train_size = int(self.train_size * len(client_data_indices[client_index]))

        train_indices = client_data_indices[client_index][:train_size]
        validation_indices = client_data_indices[client_index][train_size:]

        train_set = Subset(train_dataset, train_indices)
        validation_set = Subset(train_dataset, validation_indices)

        return self._build_dataloaders(
            batch_size=self.batch_size,
            train=train_set,
            validation=validation_set,
            test=test_dataset,
        )

    def get_dataloaders(self, settings: DatasetSettings) -> DataLoaders:
        RandomUtils.set_randomness(seed=settings.seed)
        if isinstance(settings, IidDatasetSettings):
            return self._build_dataloaders_iid(settings=settings)
        if isinstance(settings, NonIidDirichletDatasetSettings):
            return self._build_dataloaders_non_iid_dirichlet(settings=settings)
        if isinstance(settings, NonIidNonOverlappingClassesDatasetSettings):
            return self._build_dataloaders_non_iid(settings=settings)
        raise NotImplementedError
