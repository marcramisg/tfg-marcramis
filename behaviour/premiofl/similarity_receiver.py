from datetime import datetime, timezone
from typing import TYPE_CHECKING

import asyncio

from aioxmpp import JID
from spade.behaviour import CyclicBehaviour

from ...message.message import RfMessage
from ...similarity.similarity_vector import SimilarityVector

if TYPE_CHECKING:
    from ...agent.base import PremioFlAgent


class SimilarityReceiverBehaviour(CyclicBehaviour):

    def __init__(self) -> None:
        self.agent: "PremioFlAgent"
        super().__init__()

    async def run(self) -> None:
        if not self.agent.presence.is_available():
            await asyncio.sleep(0.5)
            return
        timeout = 5
        msg = await self.agent.receive(self, timeout=timeout)
        if msg and RfMessage.is_completed(message=msg) and not self.agent.are_max_iterations_reached():
            self.agent.message_logger.log(
                current_round=self.agent.current_round,
                sender=str(msg.sender.bare()),
                to=str(msg.to.bare()),
                msg_type="RECV-SIMILARITY",
                size=len(msg.body),
                thread=msg.thread,
            )
            vector = SimilarityVector.from_message(message=msg)
            vector.received_time_z = datetime.now(tz=timezone.utc)  # zulu = utc+0

            if not vector.sent_time_z:
                error_msg = (
                    f"[{self.agent.current_round}] Similarity vector from {msg.sender.bare()} without " + "timestamp."
                )
                self.agent.logger.exception(error_msg)
                raise ValueError(error_msg)

            self.agent.similarity_manager.add_similarity_vector(neighbour=msg.sender, vector=vector, thread=msg.thread)

            seconds_since_message_sent = vector.received_time_z - vector.sent_time_z
            self.agent.logger.debug(
                f"[{self.agent.current_round}] Similarity vector ({msg.thread}) received from "
                + f"{msg.sender.bare()} in SimilarityReceiverBehaviour with time elapsed "
                + f"{seconds_since_message_sent.total_seconds():.2f}"
            )

            if vector.request_reply:
                reply_vector = self.agent.similarity_manager.get_own_similarity_vector()
                if not reply_vector:
                    raise RuntimeError("Trying to compute the similarity vector without similarity function.")
                reply_vector.owner = self.agent.jid
                await self.send_similarity_vector(thread=msg.thread, vector=reply_vector, neighbour=msg.sender)
                self.agent.logger.debug(
                    f"[{self.agent.current_round}] Similarity vector ({msg.thread}) sent to "
                    + f"{msg.sender.bare()} because it is an answer to request reply."
                )

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
