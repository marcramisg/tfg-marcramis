import random
from typing import OrderedDict

from aioxmpp import JID
from torch import Tensor

from ...datatypes.consensus_manager import ConsensusManager
from ...datatypes.models import ModelManager
from ...similarity.similarity_manager import SimilarityManager
from ...similarity.similarity_vector import SimilarityVector
from ..agent import PremioFlAgent


class MacoflAgent(PremioFlAgent):

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
        result: dict[JID, OrderedDict[str, Tensor]] = {}
        for n in neighbours_vectors.keys():
            layers: OrderedDict[str, Tensor] = OrderedDict()
            layer_name = random.choice(list(self.model_manager.model.state_dict().keys()))
            layers[layer_name] = self.model_manager.model.state_dict()[layer_name]
            result[n] = layers
        return result
