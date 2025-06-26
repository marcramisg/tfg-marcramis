from abc import ABCMeta, abstractmethod
from typing import Any

from spade.message import Message
from five_client.data.camera import ImageData


class MessageParser(metaclass=ABCMeta):
    """
    Interface for message parsing
    """

    @abstractmethod
    def parse(self, message: Message) -> Any:
        pass


class ImageParser(MessageParser):
    """
    Parses base64 encoded image data
    """

    def parse(self, message: Message) -> ImageData:
        return ImageData(message.body)


class DefaultParser(MessageParser):
    """
    Parses a simple SPADE message
    """

    def parse(self, message: Message) -> Message:
        # No specific parsing needed for default SPADE messages
        return message
