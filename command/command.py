from abc import ABCMeta, abstractmethod


class Command(metaclass=ABCMeta):
    """
    Interface for commands
    """

    @abstractmethod
    async def execute(self) -> None:
        pass
