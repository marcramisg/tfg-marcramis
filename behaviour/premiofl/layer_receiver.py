from datetime import datetime, timezone
from typing import TYPE_CHECKING, OrderedDict

import asyncio
from aioxmpp import JID
from spade.behaviour import CyclicBehaviour
from torch import Tensor

from ...datatypes.consensus import Consensus
from ...message.message import RfMessage

if TYPE_CHECKING:
    from ...agent.base import PremioFlAgent


class LayerReceiverBehaviour(CyclicBehaviour):

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
                sender=msg.sender,
                to=msg.to,
                msg_type="RECV-LAYERS",
                size=len(msg.body),
                thread=msg.thread,
            )
            consensus_tr = Consensus.from_message(message=msg)
            consensus_tr.sender = msg.sender.bare()
            consensus_tr.received_time_z = datetime.now(tz=timezone.utc)  # zulu = utc+0

            if not consensus_tr.sent_time_z:
                error_msg = (
                    f"[{self.agent.current_round}] Consensus message from {msg.sender.bare()} without " + "timestamp."
                )
                self.agent.logger.exception(error_msg)
                raise ValueError(error_msg)

            time_elapsed = consensus_tr.received_time_z - consensus_tr.sent_time_z
            max_seconds_consensus = self.agent.consensus_manager.max_seconds_to_accept_consensus

            if time_elapsed.total_seconds() <= max_seconds_consensus:
                self.agent.logger.debug(
                    f"[{self.agent.current_round}] Consensus message accepted in LayerReceiverBehaviour with "
                    + f"time elapsed {time_elapsed.total_seconds():.2f}"
                )
                self.agent.consensus_manager.add_consensus(consensus=consensus_tr, thread=msg.thread)

            else:
                self.agent.logger.debug(
                    f"[{self.agent.current_round}] Consensus message discarted in LayerReceiverBehaviour because"
                    + f" time elapsed is {time_elapsed.total_seconds():.2f} and maximum is {max_seconds_consensus:.2f}"
                )

            if not self.agent.model_manager.is_training():
                # Send consensus messages that require my response
                pending_to_send = self.agent.consensus_manager.prepare_replies_to_send(
                    # sender=self.agent.jid.bare()
                )
                for consensus, thread in pending_to_send:
                    layers = self.agent.model_manager.get_layers(layers=list(consensus.layers.keys()))
                    await self.send_layers(neighbour=consensus.sender, layers=layers, thread=thread)

    async def send_layers(
        self,
        neighbour: JID,
        layers: OrderedDict[str, Tensor],
        thread: None | str = None,
    ) -> None:
        metadata = {"rf.conversation": "layers"}
        await self.agent.send_local_layers(
            neighbour=neighbour,
            request_reply=False,
            layers=layers,
            thread=thread,
            metadata=metadata,
            behaviour=self,
        )
        self.agent.logger.debug(
            f"[{self.agent.current_round}] Sent to {neighbour.localpart} the layers: {list(layers.keys())}."
        )
