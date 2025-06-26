import asyncio
import logging

from spade.behaviour import CyclicBehaviour


class ObserverBehaviour(CyclicBehaviour):

    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        super().__init__()

    async def run(self) -> None:
        await asyncio.sleep(0.1)
        msg = await self.receive(1)
        if msg:
            self.logger.info(msg.body)