import asyncio
from datetime import datetime, timedelta, timezone
from queue import Queue
from typing import OrderedDict

from aioxmpp import JID
from torch import Tensor

from ..datatypes.models import ModelManager
from .consensus import Consensus


# NOTE: Make a Manager(Abstract) that can store data, waiting data, etc. of a Generic type T
# then ConsensusManager(Manager) and T would be the Consensus.
class ConsensusManager:

    def __init__(
        self,
        model_manager: ModelManager,
        max_order: int,
        max_seconds_to_accept_consensus: float,
        wait_for_responses_timeout: float = 2 * 60,
        epsilon_margin: float = 0.05,
        consensus_iterations: int = 1,
    ) -> None:
        self.model_manager = model_manager
        self.max_order = max_order
        self.max_seconds_to_accept_consensus = max_seconds_to_accept_consensus
        self.wait_for_responses_timeout = wait_for_responses_timeout
        self.epsilon_margin = epsilon_margin
        self.received_consensus: Queue[Consensus] = Queue()
        self.waiting_responses: dict[JID, list[str]] = (
            {}
        )  # Neighbours I am waiting for. list[str] are the layers requested to the neighbour JID.
        self.to_response: Queue[tuple[Consensus, str | None]] = (
            Queue()
        )  # Neighbours waiting to my response. [str] is the thread and [Consensus] because stores layers.
        self.max_iterations = consensus_iterations
        self.__completed_iterations: int = 0
        self.__last_algorithm_iteration: int = -1

    def prepare_replies_to_send(self) -> list[tuple[Consensus, str | None]]:
        responses: list[tuple[Consensus, str | None]] = []
        while self.to_response.qsize() > 0:
            consensus, thread = self.to_response.get()
            response = Consensus(
                layers=self.model_manager.get_layers(list(consensus.layers.keys())),
                request_reply=False,
                sender=consensus.sender,
            )
            responses.append((response, thread))
            self.to_response.task_done()
        return responses

    def add_consensus(self, consensus: Consensus, thread: None | str) -> None:
        if (
            consensus.sender
            and consensus.sender.bare() in self.waiting_responses
            and list(consensus.layers.keys()) == self.waiting_responses[consensus.sender.bare()]
        ):
            del self.waiting_responses[consensus.sender.bare()]
        elif consensus.request_reply:
            self.to_response.put((consensus, thread))
        self.received_consensus.put(consensus)

    async def wait_receive_consensus(self, timeout: None | float = None) -> bool:
        to = timeout if timeout is not None else self.wait_for_responses_timeout
        start_time_z = datetime.now(tz=timezone.utc)
        stop_time_reached = False
        while self.waiting_responses and not stop_time_reached:
            await asyncio.sleep(delay=2)
            stop_time_z = datetime.now(tz=timezone.utc) + timedelta(seconds=to)
            stop_time_reached = stop_time_z >= start_time_z
        return len(list(self.waiting_responses.keys())) == 0

    def apply_consensus(self, consensus: Consensus) -> None:
        if self.model_manager.is_training():
            raise RuntimeError("Trying to apply consensus while training the model.")
        consensuated_model = ConsensusManager.apply_consensus_to_layers(
            full_model=self.model_manager.model.state_dict(),
            layers=consensus.layers,
            max_order=self.max_order,
            epsilon_margin=self.epsilon_margin,
        )
        self.model_manager.replace_all_layers(new_layers=consensuated_model)

    def apply_all_consensus(
        self,
    ) -> list[Consensus]:
        consumed_consensus_transmissions: list[Consensus] = []
        while self.received_consensus.qsize() > 0:
            ct = self.received_consensus.get()
            ct.processed_start_time_z = datetime.now(tz=timezone.utc)
            self.apply_consensus(ct)
            ct.processed_end_time_z = datetime.now(tz=timezone.utc)
            consumed_consensus_transmissions.append(ct)
            self.received_consensus.task_done()
        return consumed_consensus_transmissions

    @staticmethod
    def apply_consensus_to_layers(
        full_model: OrderedDict[str, Tensor],
        layers: OrderedDict[str, Tensor],
        max_order: int = 2,
        epsilon_margin: float = 0.05,
    ) -> OrderedDict[str, Tensor]:
        consensuated_result: OrderedDict[str, Tensor] = OrderedDict()
        for key in full_model.keys():
            if key in layers:
                consensuated_result[key] = ConsensusManager.apply_consensus_to_tensors(
                    tensor_a=full_model[key],
                    tensor_b=layers[key],
                    max_order=max_order,
                    epsilon_margin=epsilon_margin,
                )
            else:
                consensuated_result[key] = full_model[key]
        return consensuated_result

    @staticmethod
    def apply_consensus_to_tensors(
        tensor_a: Tensor, tensor_b: Tensor, max_order: int, epsilon_margin: float = 0.05
    ) -> Tensor:
        """
        Computes a new consensuated `pytorch.Tensor` without modifying the input tensors.

        Args:
            tensor_a (Tensor): Input `torch.Tensor` that will be multiplied by epsilon.
            tensor_b (Tensor): Input `torch.Tensor` that will be multiplied by (1 - epsilon).
            max_order (int): Maximum order of the graph network.
            epsilon_margin (float, optional): A margin to be sure that epsilon < 1 / max_graph_degree. Defaults to 0.05.

        Raises:
            ValueError: If `max_order` is lower than 2.

        Returns:
            Tensor: The resulting Tensor after consensus.
        """
        if max_order <= 1:
            raise ValueError(f"Max order of consensus must be greater than 1 and it is {max_order}.")
        # epsilon_margin because must be LESS than 1 / max_order
        epsilon = 1 / max_order - epsilon_margin
        return epsilon * tensor_a + (1 - epsilon) * tensor_b

    def add_one_completed_iteration(self, algorithm_rounds: int) -> int:
        if algorithm_rounds != self.__last_algorithm_iteration:
            self.__last_algorithm_iteration = algorithm_rounds
            self.__completed_iterations = 0
        self.__completed_iterations += 1
        return self.__completed_iterations

    def are_max_iterations_reached(self) -> bool:
        return self.__completed_iterations >= self.max_iterations

    def get_completed_iterations(self, algorithm_rounds: int) -> int:
        if algorithm_rounds != self.__last_algorithm_iteration:
            self.__last_algorithm_iteration = algorithm_rounds
            self.__completed_iterations = 0
        return self.__completed_iterations
