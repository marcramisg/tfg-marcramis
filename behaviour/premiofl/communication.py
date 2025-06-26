import traceback
import uuid
from typing import TYPE_CHECKING, OrderedDict

import asyncio
from aioxmpp import JID
from spade.behaviour import State
from torch import Tensor

from ...similarity.similarity_vector import SimilarityVector

if TYPE_CHECKING:
    from ...agent.base import PremioFlAgent


class CommunicationState(State):
    def __init__(self) -> None:
        self.agent: "PremioFlAgent"
        super().__init__()

    async def on_start(self) -> None:
        consensus_it_id = self.agent.consensus_manager.get_completed_iterations(self.agent.current_round) + 1
        self.agent.logger.debug(f"[{self.agent.current_round}] ({consensus_it_id}) Starting CommunicationState...")

    async def run(self) -> None:
        try:
            if not self.agent.presence.is_available():
                await asyncio.sleep(0.5)
                self.set_next_state("train")
            consensus_it_id = self.agent.consensus_manager.get_completed_iterations(self.agent.current_round) + 1
            selected_neighbours = self.agent.select_neighbours()
            if selected_neighbours:
                self.agent.logger.debug(
                    f"[{self.agent.current_round}] ({consensus_it_id}) Selected neighbours of CommunicationState: "
                    + f"{[jid.localpart for jid in selected_neighbours]}"
                )

                if self.agent.similarity_manager.function is not None:
                    all_responses_received = await self.similarity_vector_exchange(neighbours=selected_neighbours)
                    self.agent.logger.info(
                        f"[{self.agent.current_round}] ({consensus_it_id}) Vector exchange completed "
                        + f"{'with all responses' if all_responses_received else 'by timeout'}."
                    )

                sent_layers = False
                for n, ls in self.agent.assign_layers(selected_neighbours).items():
                    await self.send_layers(neighbour=n, layers=ls) # añadido
                    self.agent.logger.debug(
                        f"[{self.agent.current_round}] ({consensus_it_id}) Consensus layers of CommunicationState: "
                        + f"{[jid.localpart for jid in selected_neighbours]} -> {list(ls.keys())}"
                    )
                    sent_layers = True

                if sent_layers:
                    self.agent.logger.info(
                        f"[{self.agent.current_round}] ({consensus_it_id}) Selected layers sent to selected neighbours."
                    )
                self.set_next_state("consensus")

            else:
                self.agent.logger.info(
                    f"[{self.agent.current_round}] ({consensus_it_id}) No neighbour selected in CommunicationState. "
                    + "Going to train again..."
                )
                self.set_next_state("train")

        except Exception as e:
            self.agent.logger.exception(e)
            traceback.print_exc()

    async def send_layers(
        self,
        neighbour: JID,
        layers: OrderedDict[str, Tensor],
    ) -> None:
        metadata = {"rf.conversation": "layers"}
        thread = str(uuid.uuid4())
        await self.agent.send_local_layers(
            neighbour=neighbour,
            request_reply=True,
            layers=layers,
            thread=thread,
            metadata=metadata,
            behaviour=self,
        )
        self.agent.logger.debug(
            f"[{self.agent.current_round}] Message sent to {neighbour.localpart} with the layers: {list(layers.keys())}."
        )

    async def similarity_vector_exchange(self, neighbours: list[JID]) -> bool:
        vector = self.agent.similarity_manager.get_own_similarity_vector()
        if not vector:
            raise ValueError("Trying to do the similarity vector phase without similarity function.")
        thread = str(uuid.uuid4())
        vector.owner = self.agent.jid
        vector.request_reply = True
        self.agent.similarity_manager.clear_waiting_responses(neighbours, thread)
        for neighbour in neighbours:
            await self.send_similarity_vector(thread=thread, vector=vector, neighbour=neighbour)
        try:
            return await asyncio.wait_for(self.agent.similarity_manager.wait_similarity_vectors(), timeout=5.0)  # añadido
        except asyncio.TimeoutError:
            self.agent.logger.warning("Timeout esperando vectores de similitud")

    async def send_similarity_vector(
        self,
        thread: str,
        vector: SimilarityVector,
        neighbour: JID,
    ) -> None:
        metadata = {"rf.conversation": "similarity"}
        await self.agent.send_similarity_vector(
            neighbour=neighbour,
            vector=vector,
            thread=thread,
            metadata=metadata,
            behaviour=self,
        )
        self.agent.logger.debug(
            f"[{self.agent.current_round}] Sent to {neighbour.localpart} the vector: {vector.to_message().body} "
            + f"with thread {thread}."
        )
