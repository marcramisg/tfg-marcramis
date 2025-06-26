from abc import ABCMeta
from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader, Subset


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


@dataclass
class DatasetSettings(object, metaclass=ABCMeta):
    iid: bool
    seed: None | int


class IidDatasetSettings(DatasetSettings):
    def __init__(
        self,
        seed: None | int,
        train_samples_percent: None | float = None,
        train_samples_absolute: None | int = None,
        test_samples_percent: None | float = None,
        test_samples_absolute: None | int = None,
    ) -> None:
        super().__init__(iid=True, seed=seed)
        self.train_samples_percent = train_samples_percent
        self.train_samples_absolute = train_samples_absolute
        self.test_samples_percent = test_samples_percent
        self.test_samples_absolute = test_samples_absolute

        if train_samples_percent is not None and train_samples_absolute is not None:
            raise ValueError("Absolute and percent must be exclusive in train.")
        if test_samples_percent is not None and test_samples_absolute is not None:
            raise ValueError("Absolute and percent must be exclusive in test.")

    def get_new_train_dataset(
        self,
        original: Subset,
    ) -> Subset:
        return self._get_new_dataset(
            original=original,
            new_percent=self.train_samples_percent,
            new_absolute=self.train_samples_absolute,
        )

    def get_new_test_dataset(
        self,
        original: Subset,
    ) -> Subset:
        return self._get_new_dataset(
            original=original,
            new_percent=self.test_samples_percent,
            new_absolute=self.test_samples_absolute,
        )

    def _get_new_dataset(
        self,
        original: Subset,
        new_percent: None | float = None,
        new_absolute: None | int = None,
    ) -> Subset:
        if new_percent is None and new_absolute is None:
            return original
        total_samples = len(original)
        wanted_samples: int
        if new_percent is not None:
            wanted_samples = int(total_samples * new_percent)
        elif new_absolute is not None:
            wanted_samples = new_absolute
        if total_samples < wanted_samples:
            raise ValueError(f"The num of samples is less than the requested: {total_samples} < {wanted_samples}")
        new_indices = np.random.choice(total_samples, wanted_samples, replace=False).tolist()
        return Subset(original, new_indices)

    def are_all_samples_selected(self) -> bool:
        full_train = self.are_all_train_samples_selected()
        full_test = self.are_all_test_samples_selected()
        return full_train and full_test

    def are_all_test_samples_selected(self) -> bool:
        return self.test_samples_percent is None and self.test_samples_absolute is None

    def are_all_train_samples_selected(self) -> bool:
        return self.train_samples_percent is None and self.train_samples_absolute is None


class NonIidDatasetSettings(DatasetSettings, metaclass=ABCMeta):

    def __init__(self, seed: None | int, num_clients: int, client_index: int) -> None:
        super().__init__(iid=False, seed=seed)
        self.num_clients = num_clients
        self.client_index = client_index


class NonIidNonOverlappingClassesDatasetSettings(NonIidDatasetSettings):
    def __init__(
        self,
        seed: None | int,
        num_clients: int,
        client_index: int,
        classes_per_client: int,
    ) -> None:
        super().__init__(seed=seed, num_clients=num_clients, client_index=client_index)
        self.classes_per_client = classes_per_client


class NonIidDirichletDatasetSettings(NonIidDatasetSettings):
    def __init__(
        self,
        seed: None | int,
        num_clients: int,
        client_index: int,
        dirichlet_alpha: float = 0.1,
    ) -> None:
        super().__init__(seed=seed, num_clients=num_clients, client_index=client_index)
        self.dirichlet_alpha = dirichlet_alpha
