from typing import Optional
import asyncio

from aioxmpp import JID
from spade.template import Template

from ..agent.agent import AgentBase
from ..behaviour.coordination import PresenceCoordinatorFSM


class CoordinatorAgent(AgentBase):

    def __init__(
        self,
        jid: str,
        password: str,
        coordinated_agents: list[JID],
        max_message_size: int,
        web_address: str = "0.0.0.0",
        web_port: int = 10000, 
        verify_security: bool = False,
    ):
        self.coordinated_agents = [] if coordinated_agents is None else coordinated_agents
        self.coordination_fsm: Optional["PresenceCoordinatorFSM"] = None
        self.ready_to_start_algorithm = False  # Flag to prevent finish before starting the algorithm
        super().__init__(
            jid,
            password,
            max_message_size,
            web_address,
            web_port,
            verify_security,
        )

    async def setup(self) -> None:
        await super().setup()
        # template = Template()
        # template.set_metadata("rf.presence", "sync")
        # self.coordination_fsm = PresenceCoordinatorFSM(self.coordinated_agents)
        # self.add_behaviour(self.coordination_fsm, template)
        

    async def behaviour(self):
        template = Template()
        template.set_metadata("rf.presence", "sync")
        self.coordination_fsm = PresenceCoordinatorFSM(self.coordinated_agents)
        self.add_behaviour(self.coordination_fsm, template)
