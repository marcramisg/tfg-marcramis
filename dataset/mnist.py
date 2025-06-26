from pathlib import Path
from typing import Optional

from torchvision import datasets, transforms
from torchvision.transforms import Compose

from .dataloader_generator import BaseDataLoaderGenerator


class MnistDataLoaderGenerator(BaseDataLoaderGenerator):
    """Data loader generator for the MNIST dataset."""

    def __init__(
        self,
        data_dir: str | Path = "royalflush_datasets/mnist",
        batch_size: int = 64,
        train_size: float = 0.8,
        transform: Optional[Compose] = None,
    ) -> None:
        """Initializes the MNIST data loader generator.

        Args:
            data_dir (str | Path): Directory where MNIST data will be stored.
            batch_size (int): Batch size for data loaders.
            train_size (float): Proportion of data to use for training.
            transform (Optional[Compose]): Transformations to apply to the data.
        """
        super().__init__(
            dataset_cls=datasets.MNIST,
            data_dir=data_dir,
            batch_size=batch_size,
            train_size=train_size,
            transform=transform or Compose([transforms.ToTensor()]),
        )
