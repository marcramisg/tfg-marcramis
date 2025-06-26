from queue import Queue

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

from five_client.parser import MessageFactory
from five_client.command import Command


class Perception(CyclicBehaviour):

    def __init__(self, queue: Queue) -> None:
        super().__init__()
        self.agent: Agent
        self.queue: Queue = queue

    async def on_start(self) -> None:
        print(f"{self.agent.name}: running perception state.")

    async def run(self) -> None:
        message: Message = await self.receive(timeout=10)
        if message:
            parser = MessageFactory.create_parser(message)
            data = parser.parse(message)
            self.queue.put(data)

    async def on_end(self) -> None:
        print(f"{self.agent.name}: stopping...")
        await self.agent.stop()


class Action(CyclicBehaviour):

    def __init__(self, queue: Queue[Command]) -> None:
        super().__init__()
        self.agent: Agent
        self.queue: Queue[Command] = queue

    async def run(self) -> None:
        if not self.queue.empty():
            command: Command = self.queue.get()
            await command.execute()
