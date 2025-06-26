import asyncio
from datetime import datetime, timedelta, timezone

from aioxmpp import JID

from ..datatypes.models import ModelManager
from .function import SimilarityFunction
from .similarity_vector import SimilarityVector


class SimilarityManager:

    def __init__(
        self,
        model_manager: ModelManager,
        wait_for_responses_timeout: float = 2 * 60,
        function: None | SimilarityFunction = None,
    ) -> None:
        # NOTE Operations such as adding, removing, and reading a value on a dict are atomic.
        # Specifically:
        #     Adding a key and value mapping.
        #     Replacing a value for a key.
        #     Adding a dict to a dict via update().
        #     Getting a list of keys via keys().
        self.model_manager = model_manager
        self.function = function
        self.wait_for_responses_timeout = wait_for_responses_timeout
        self.waiting_responses: dict[JID, str] = (
            {}
        )  # Neighbours I am waiting for. JID is the sender agent and str is the thread of the message.
        self.to_response: list[tuple[JID, str]] = (
            []
        )  # Neighbours waiting to my response. [tuple[JID, str]] are tuples of neighbours and threads.
        self.similarity_vectors: dict[JID, SimilarityVector] = {}

    def clear_waiting_responses(self, neighbours: list[JID], thread: str) -> None:
        self.waiting_responses = {n.bare(): thread for n in neighbours}

    def get_own_similarity_vector(self) -> SimilarityVector | None:
        if self.function is None:
            # raise ValueError(
            #     "The agent must have a function to compute the similarity vector."
            # )
            return None
        layer2 = self.model_manager.model.state_dict()
        vector = self.function.get_similarity_vector(
            layers1=self.model_manager.initial_state,
            layers2=layer2,
        )
        vector.sent_time_z = datetime.now(tz=timezone.utc)
        return vector

    async def wait_similarity_vectors(self, timeout: None | float = None) -> bool:
        to = timeout if timeout is not None else self.wait_for_responses_timeout
        stop_time_z = datetime.now(tz=timezone.utc) + timedelta(seconds=to)
        stop_time_reached = False
        while list(self.waiting_responses.keys()) and not stop_time_reached:
            await asyncio.sleep(delay=2)
            stop_time_reached = datetime.now(tz=timezone.utc) >= stop_time_z
        return len(list(self.waiting_responses.keys())) == 0

    def add_similarity_vector(self, neighbour: JID, vector: SimilarityVector, thread: None | str) -> None:
        if thread and neighbour.bare() in self.waiting_responses and thread == self.waiting_responses[neighbour.bare()]:
            del self.waiting_responses[neighbour.bare()]
        self.similarity_vectors[neighbour.bare()] = vector

    def get_vector(self, neighbour: JID) -> SimilarityVector | None:
        if neighbour.bare() in self.similarity_vectors:
            return self.similarity_vectors[neighbour.bare()]
        return None
