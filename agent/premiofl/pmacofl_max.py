import random
from typing import OrderedDict

from aioxmpp import JID
from torch import Tensor

from ...datatypes.consensus_manager import ConsensusManager
from ...datatypes.models import ModelManager
from ...similarity.similarity_manager import SimilarityManager
from ...similarity.similarity_vector import SimilarityVector
from ..agent import PremioFlAgent


class PmacoflMaxAgent(PremioFlAgent):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        consensus_manager: ConsensusManager,
        model_manager: ModelManager,
        similarity_manager: SimilarityManager,
        observers: list[JID] | None = None,
        neighbours: list[JID] | None = None,
        coordinator: JID | None = None,
        max_rounds: int | None = 100,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        super().__init__(
            jid,
            password,
            max_message_size,
            consensus_manager,
            model_manager,
            similarity_manager,
            observers,
            neighbours,
            coordinator,
            max_rounds,
            web_address,
            web_port,
            verify_security,
        )

    def _select_neighbours(self, neighbours: list[JID]) -> list[JID]:
        if not neighbours:
            return []
        return [random.choice(neighbours)]

    def _assign_layers(
        self,
        my_vector: None | SimilarityVector,
        neighbours_vectors: dict[JID, SimilarityVector],
        selected_neighbours: list[JID],
    ) -> dict[JID, OrderedDict[str, Tensor]]:
        if not my_vector:
            raise ValueError("PMACoFL algorithms must have a similarity function.")
        result: dict[JID, OrderedDict[str, Tensor]] = {}

        for neighbour in selected_neighbours:
            neighbour_vector = neighbours_vectors[neighbour].vector

            max_layer: str | None = None
            max_difference = float("-inf")

            for layer_name, my_layer_value in my_vector.vector.items():
                neighbour_layer_value = neighbour_vector[layer_name]

                difference = abs(my_layer_value - neighbour_layer_value)

                if difference > max_difference:
                    max_difference = difference
                    max_layer = layer_name

            if max_layer:
                layer_tensor = self.model_manager.model.state_dict()[max_layer]
                result[neighbour] = OrderedDict({max_layer: layer_tensor})

        return result
