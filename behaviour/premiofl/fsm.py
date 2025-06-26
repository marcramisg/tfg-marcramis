from typing import TYPE_CHECKING

from spade.behaviour import FSMBehaviour

from .communication import CommunicationState
from .consensus import ConsensusState
from .train import TrainAndApplyConsensusState

if TYPE_CHECKING:
    from ...agent.base import PremioFlAgent


class PremioFsmBehaviour(FSMBehaviour):

    def __init__(self) -> None:
        self.agent: "PremioFlAgent"
        self.train_state = TrainAndApplyConsensusState()
        self.send_state = CommunicationState()
        self.consensus_state = ConsensusState()
        super().__init__()

    def setup(self) -> None:
        self.add_state(name="train", state=self.train_state, initial=True)
        self.add_state(name="communication", state=self.send_state)
        self.add_state(name="consensus", state=self.consensus_state)
        self.add_transition(source="train", dest="communication")
        self.add_transition(source="communication", dest="train")
        self.add_transition(source="communication", dest="consensus")
        self.add_transition(source="consensus", dest="communication")
        self.add_transition(source="consensus", dest="train")

    async def on_start(self) -> None:
        self.agent.logger.debug("FSM algorithm started.")

    async def on_end(self) -> None:
        self.agent.logger.debug("FSM algorithm finished.")
