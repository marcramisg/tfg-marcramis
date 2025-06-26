from typing import Optional

from aioxmpp import JID
from spade.template import Template

from five_client.agent import AgentBase
from ..behaviour.observer import ObserverBehaviour


class ObserverAgent(AgentBase):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.agents_observed: list[JID] = []
        self.observation_theme_behaviours: dict[str, Optional[ObserverBehaviour]] = {
            "message": None,
            "nn": None,
            "iteration": None,
        }
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
        for theme in self.observation_theme_behaviours:
            template = Template(metadata={"rf.observe": theme})
            behaviour = ObserverBehaviour(f"rf.{theme}.{self.name}")
            self.observation_theme_behaviours[theme] = behaviour
            self.add_behaviour(behaviour, template)