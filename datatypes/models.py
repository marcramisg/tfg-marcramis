import codecs
import copy
import pickle
from datetime import datetime, timezone
from typing import Callable, OrderedDict

import random
import time
import torch
from aioxmpp import JID
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..datatypes.metrics import ModelMetrics

# from ..utils.random import RandomUtils
from .data import DataLoaders


class ModelManager:
    """
    Handles the Neural Network model training, validation and testing.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: _Loss,
        optimizer: Optimizer,
        batch_size: int,
        training_epochs: int,
        dataloaders: DataLoaders,
        device: None | str = None,
        seed: None | int = 42,
        deterministic: bool = False,
        track_layers_weights: None | list[str] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.dataloaders = dataloaders
        self.seed = seed
        self.deterministic = deterministic
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self.model = self.model.to(self.device)
        self.__move_optimizer_to_device()
        self.track_layers_weights: list[str] = [] if track_layers_weights is None else track_layers_weights
        # NOTE when the below NOTE is completed, uncomment: RandomUtils.set_randomness(seed=self.seed)
        # NOTE Ask for a model generator and generate the model here: self.model = generator.get_model(parameters)
        self.initial_state: OrderedDict[str, Tensor] = copy.deepcopy(model.state_dict())
        # self.pretrain_state: OrderedDict[str, Tensor] = copy.deepcopy(
        #     self.model.state_dict()
        # )
        self.__training: bool = False

    def is_training(self) -> bool:
        return self.__training

    def replace_all_layers(self, new_layers: OrderedDict[str, Tensor]) -> None:
        self.model.load_state_dict(state_dict=new_layers)

    def train(
        self,
        epochs: None | int = None,
        train_logger: None | Callable[[int, ModelMetrics, JID, int], None] = None,
        weight_logger: None | Callable[[int, str, float, JID, int], None] = None,
        agent_jid: None | JID = None,
        current_round: None | int = None,
    ) -> list[ModelMetrics]:
        """
        Updates the model by training on the training dataset and optionally tracks specific weights.
        Args:
            epochs: Number of epochs to train.
            train_logger: Logger function for metrics.
            weight_logger: Logger function for track weight convergence given a layer name.
            agent_jid: Agent's JID for logging.
            current_round: Current algorithm round.
        """
        # self.pretrain_state = copy.deepcopy(self.model.state_dict())
        self.__training = True
        if epochs is None:
            epochs = self.training_epochs

        metrics: list[ModelMetrics] = []

        # Training loop
        try:
            for epoch in range(epochs):
                # Log current weight state
                if weight_logger is not None and agent_jid is not None and current_round is not None:
                    current_state = self.model.state_dict()
                    for layer in self.track_layers_weights:
                        first_weight = float(current_state[layer].flatten()[0].item())
                        weight_logger(
                            epoch + 1,
                            layer,
                            first_weight,
                            agent_jid,
                            current_round,
                        )

                # Start training
                self.model.train()
                total_loss: float = 0.0
                correct: int = 0
                total_samples: int = 0

                images: Tensor
                labels: Tensor
                outputs: Tensor
                loss: Tensor
                predicted: Tensor

                init_time_z = datetime.now(tz=timezone.utc)
                for images, labels in self.dataloaders.train:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    correct += int((predicted == labels).sum().item())

                # Log metrics after each epoch
                epoch_metric: ModelMetrics = ModelMetrics(
                    accuracy=(correct / total_samples),
                    loss=(total_loss / len(self.dataloaders.train)),
                    start_time_z=init_time_z,
                    end_time_z=datetime.now(tz=timezone.utc),
                )
                metrics.append(epoch_metric)
                if train_logger is not None and agent_jid is not None and current_round is not None:
                    train_logger(epoch + 1, epoch_metric, agent_jid, current_round)

            return metrics

        finally:
            self.__training = False

    def _inference(self, dataloader: DataLoader) -> ModelMetrics:
        """
        Performs inference on a given dataset and returns metrics.
        """

        # Validation
        self.model.eval()
        correct: int = 0
        total: int = 0
        predicted_labels: list[int] = []
        true_labels: list[int] = []
        total_loss: float = 0.0

        images: Tensor
        labels: Tensor
        outputs: Tensor
        loss: Tensor
        predicted: Tensor

        init_time_z = datetime.now(tz=timezone.utc)
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += int((predicted == labels).sum().item())
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        end_time_z: datetime = datetime.now(tz=timezone.utc)
        accuracy: float = correct / total
        resulting_loss: float = total_loss / len(dataloader)
        metrics: ModelMetrics = ModelMetrics(
            accuracy=accuracy,
            loss=resulting_loss,
            start_time_z=init_time_z,
            end_time_z=end_time_z,
        )
        return metrics

    def inference(self) -> ModelMetrics:
        """
        Returns the TRAIN inference metrics using validation data.
        """
        return self._inference(dataloader=self.dataloaders.validation)

    def test_inference(self) -> ModelMetrics:
        """
        Returns the TEST inference metrics.
        """
        return self._inference(dataloader=self.dataloaders.test)

    def get_layers(self, layers: list[str], deepcopy_layers: bool = False) -> OrderedDict[str, Tensor]:
        selected_layers: OrderedDict[str, Tensor] = OrderedDict()
        for layer in layers:
            if deepcopy_layers:
                selected_layers[layer] = copy.deepcopy(self.model.state_dict()[layer])
            else:
                selected_layers[layer] = self.model.state_dict()[layer]
        return selected_layers

    @staticmethod
    def export_layers(layers: OrderedDict[str, Tensor]) -> str:
        return codecs.encode(pickle.dumps(layers), encoding="base64").decode(encoding="utf-8")

    @staticmethod
    def import_layers(
        base64_codified_layers: str,
    ) -> OrderedDict[str, Tensor]:
        return pickle.loads(codecs.decode(base64_codified_layers.encode(encoding="utf-8"), encoding="base64"))

    def save_model_to_file(self, filepath: str) -> None:
        """
        Saves the model into a file.

        Args:
            filepath (str): The path to the file where the model will be saved.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model_from_file(self, filepath: str) -> None:
        """
        Loads the model from a file.

        Args:
            filepath (str): The path to the file from which to load the model.
        """
        self.model.load_state_dict(torch.load(filepath))

    def __move_optimizer_to_device(self) -> None:
        param: Parameter
        self.model = self.model.to(self.device)
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.device != self.device:
                    param.data = param.data.to(self.device)
                    if param.grad is not None:
                        param.grad = param.grad.to(self.device)


class DummyModelManager:
    def __init__(
        self,
        model: nn.Module,
        criterion: _Loss,
        optimizer: Optimizer,
        batch_size: int,
        training_epochs: int,
        dataloaders: DataLoaders,
        device: None | str = None,
        seed: None | int = 42,
        deterministic: bool = False,
        track_layers_weights: None | list[str] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.dataloaders = dataloaders
        self.seed = seed
        self.deterministic = deterministic
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self.model = self.model.to(self.device)
        self.__move_optimizer_to_device()
        self.track_layers_weights: list[str] = [] if track_layers_weights is None else track_layers_weights
        # NOTE when the below NOTE is completed, uncomment: RandomUtils.set_randomness(seed=self.seed)
        # NOTE Ask for a model generator and generate the model here: self.model = generator.get_model(parameters)
        self.initial_state: OrderedDict[str, Tensor] = copy.deepcopy(model.state_dict())
        # self.pretrain_state: OrderedDict[str, Tensor] = copy.deepcopy(
        #     self.model.state_dict()
        # )
        self.__training: bool = False
        self.current_accuracy = 0.0
        self.current_loss = 2.5

    def is_training(self) -> bool:
        return self.__training

    def replace_all_layers(self, new_layers: OrderedDict[str, Tensor]) -> None:
        pass  # Simulación: no hace nada

    def train(
        self,
        epochs: None | int = None,
        train_logger: None | Callable[[int, ModelMetrics, JID, int], None] = None,
        weight_logger: None | Callable[[int, str, float, JID, int], None] = None,
        agent_jid: None | JID = None,
        current_round: None | int = None,
    ) -> list[ModelMetrics]:
        self.__training = True
        time.sleep(1.5)
        if epochs is None:
            epochs = self.training_epochs

        metrics = []
        for epoch in range(epochs):
            metric = self._generate_fake_metric()

            metrics.append(metric)

            if train_logger and agent_jid and current_round is not None:
                train_logger(epoch + 1, metric, agent_jid, current_round)

            if weight_logger and agent_jid and current_round is not None:
                for layer in self.track_layers_weights:
                    dummy_weight = random.uniform(-1.0, 1.0)
                    weight_logger(epoch + 1, layer, dummy_weight, agent_jid, current_round)
            time.sleep(2)

        self.__training = False
        return metrics

    def _generate_fake_metric(self) -> ModelMetrics:
        accuracy_increment = random.uniform(0.0, 0.10)
        loss_decrement = random.uniform(0.0, 0.1)

        self.current_accuracy = min(1.0, self.current_accuracy + accuracy_increment)
        self.current_loss = max(0.0, self.current_loss - loss_decrement)
        
        return ModelMetrics(
            accuracy=self.current_accuracy,
            loss=self.current_loss,
            start_time_z=datetime.now(tz=timezone.utc),
            end_time_z=datetime.now(tz=timezone.utc),
        )

    def _inference(self, dataloader) -> ModelMetrics:
        return self._generate_fake_metric()

    def inference(self) -> ModelMetrics:
        return self._generate_fake_metric()

    def test_inference(self) -> ModelMetrics:
        return self._generate_fake_metric()

    def get_layers(self, layers: list[str], deepcopy_layers: bool = False) -> OrderedDict[str, Tensor]:
        dummy_tensor = torch.tensor(0.0)
        return OrderedDict((layer, dummy_tensor.clone()) for layer in layers)

    @staticmethod
    def export_layers(layers: OrderedDict[str, Tensor]) -> str:
        return ""

    @staticmethod
    def import_layers(base64_codified_layers: str) -> OrderedDict[str, Tensor]:
        return OrderedDict()

    def save_model_to_file(self, filepath: str) -> None:
        pass  # No guarda nada

    def load_model_from_file(self, filepath: str) -> None:
        pass  # No carga nada

    def __move_optimizer_to_device(self) -> None:
        pass  # No hay optimizer real