import random
from typing import OrderedDict

from aioxmpp import JID
from torch import Tensor

from royalflush.similarity.similarity_vector import SimilarityVector

from ..agent import PremioFlAgent


class AcolAgent(PremioFlAgent):

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
        return {n: self.model_manager.model.state_dict() for n in selected_neighbours}
