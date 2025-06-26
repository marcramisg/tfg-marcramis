import json

from abc import abstractmethod, ABCMeta
from typing import Union

from aioxmpp import JID
from spade.message import Message
from spade.behaviour import CyclicBehaviour
from spade.agent import Agent

from five_client.command import Command
from five_client.data import PositionData, Rgba


class FiveCommand(Command, metaclass=ABCMeta):
    """
    Interface for FIVE commands
    """

    def __init__(
        self,
        name: str,
        fiveserver_jid: JID,
        sender_state: CyclicBehaviour,
    ) -> None:
        super().__init__()
        self.name: str = name
        self.fiveserver_jid: JID = fiveserver_jid
        self.sender_state: CyclicBehaviour = sender_state

    async def execute(self) -> None:
        message = self.build_message()
        await self.sender_state.send(msg=message)

    def build_message(self) -> Message:
        """
        Builds a spade.message.Message representation of the command
        to send the command to the FIVE server.

        Returns:
            Message: The resulting Message to send to the FIVE server.
        """
        agent: Agent = self.sender_state.agent
        command = {"commandName": self.name, "data": self.data()}
        message = Message(metadata={"five": "command"})
        message.sender = str(agent.jid)
        message.to = str(self.fiveserver_jid)
        message.body = json.dumps(obj=command)
        return message

    @property
    @abstractmethod
    def data(self) -> list[str]:
        pass


class CreateAgent(FiveCommand):

    def __init__(
        self,
        fiveserver_jid: JID,
        sender_state: CyclicBehaviour,
        agent_nickname: str,
        agent_type: str,
        starter_position: Union[str, PositionData],
    ) -> None:
        self.agent_nickname: str = agent_nickname
        self.agent_type: str = agent_type
        self.starter_position: Union[str, PositionData] = starter_position
        super().__init__(
            name="create",
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
        )

    @property
    def data(self) -> list[str]:
        return [
            self.agent_nickname,
            self.agent_type,
            str(self.starter_position),  # str(position) is a json.dumps(position)
        ]


class Move(FiveCommand):

    def __init__(
        self, fiveserver_jid: JID, sender_state: CyclicBehaviour, position: PositionData
    ) -> None:
        self.position: PositionData = position
        super().__init__(
            name="moveTo",
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
        )

    @property
    def data(self) -> list[str]:
        return [str(self.position)]  # str(position) is a json.dumps(position)


class ChangeColor(FiveCommand):

    def __init__(
        self, fiveserver_jid: JID, sender_state: CyclicBehaviour, color: Rgba
    ) -> None:
        self.color: Rgba = color
        super().__init__(
            name="color",
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
        )

    @property
    def data(self) -> list[str]:
        return [str(self.color)]  # str(color) is a json.dumps(color)
