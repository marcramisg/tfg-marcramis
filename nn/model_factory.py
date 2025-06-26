from torch import nn
from torch.optim import Adam

from ..dataset.cifar import Cifar10DataLoaderGenerator, Cifar100DataLoaderGenerator
from ..dataset.dataloader_generator import BaseDataLoaderGenerator
from ..dataset.mnist import MnistDataLoaderGenerator
from ..datatypes.data import DatasetSettings
from ..datatypes.models import ModelManager, DummyModelManager
from ..utils.random import RandomUtils
from .model.cnn import CNN5
from .model.mlp import CifarMlp


class ModelManagerFactory:

    @staticmethod
    def get_dataloader_generator(dataset: str) -> BaseDataLoaderGenerator:
        if dataset == "cifar10":
            return Cifar10DataLoaderGenerator()
        if dataset == "cifar100":
            return Cifar100DataLoaderGenerator()
        if dataset == "mnist":
            return MnistDataLoaderGenerator()
        raise NotImplementedError(f"Dataset {dataset} is not valid.")

    @staticmethod
    def get_ann(dataset: str, ann: str) -> nn.Module:
        input_dim, out_classes = None, None

        # Datasets
        if dataset == "cifar10":
            input_dim = (3, 32, 32)
            out_classes = 10
        if dataset == "cifar100":
            input_dim = (3, 32, 32)
            out_classes = 100
        if dataset == "mnist":
            input_dim = (1, 28, 28)
            out_classes = 10
        if input_dim is None or out_classes is None:
            raise RuntimeError("The dimension of the input/output of the ANN must be declared")

        # ANNs
        if ann == "cnn5":
            return CNN5(input_dim=input_dim, out_classes=out_classes)
        if ann == "cifar_mlp":
            return CifarMlp(input_dim=input_dim, out_classes=out_classes)
        raise NotImplementedError(f"ANN {ann} is not valid.")

    @staticmethod
    def get_manager(
        dataset: str,
        settings: DatasetSettings,
        ann: str,
        training_epochs: int,
        dummy: bool = False,
    ) -> ModelManager:
        generator = ModelManagerFactory.get_dataloader_generator(dataset=dataset)
        dataloaders = generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = ModelManagerFactory.get_ann(dataset=dataset, ann=ann)
        if dummy:
            return DummyModelManager(
                model=model,
                criterion=nn.CrossEntropyLoss(),
                optimizer=Adam(
                    model.parameters(),
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    eps=1e-7,
                    weight_decay=0,
                    amsgrad=False,
                ),
                batch_size=64,
                training_epochs=training_epochs,
                dataloaders=dataloaders,
                seed=settings.seed,
                track_layers_weights=list(model.state_dict().keys()),
                )

        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0,
                amsgrad=False,
            ),
            batch_size=64,
            training_epochs=training_epochs,
            dataloaders=dataloaders,
            seed=settings.seed,
            track_layers_weights=list(model.state_dict().keys()),
        )

    @staticmethod
    def get_cifar10_mlp(settings: DatasetSettings) -> ModelManager:
        cifar10_generator = Cifar10DataLoaderGenerator()
        dataloaders = cifar10_generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = CifarMlp(input_dim=(3, 32, 32), out_classes=10)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0,
                amsgrad=False,
            ),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=settings.seed,
            track_layers_weights=list(model.state_dict().keys()),
        )

    @staticmethod
    def get_cifar100_mlp(settings: DatasetSettings) -> ModelManager:
        cifar100_generator = Cifar100DataLoaderGenerator()
        dataloaders = cifar100_generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = CifarMlp(input_dim=(3, 32, 32), out_classes=100)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0,
                amsgrad=False,
            ),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=settings.seed,
            track_layers_weights=list(model.state_dict().keys()),
        )

    @staticmethod
    def get_cifar10_cnn5(settings: DatasetSettings) -> ModelManager:
        cifar10_generator = Cifar10DataLoaderGenerator()
        dataloaders = cifar10_generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = CNN5(input_dim=(3, 32, 32), out_classes=10)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0,
                amsgrad=False,
            ),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=settings.seed,
            track_layers_weights=list(model.state_dict().keys()),
        )

    @staticmethod
    def get_cifar100_cnn5(settings: DatasetSettings) -> ModelManager:
        cifar100_generator = Cifar100DataLoaderGenerator()
        dataloaders = cifar100_generator.get_dataloaders(settings=settings)
        RandomUtils.set_randomness(seed=settings.seed)
        model = CNN5(input_dim=(3, 32, 32), out_classes=100)
        return ModelManager(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=0,
                amsgrad=False,
            ),
            batch_size=64,
            training_epochs=1,
            dataloaders=dataloaders,
            seed=settings.seed,
            track_layers_weights=list(model.state_dict().keys()),
        )
