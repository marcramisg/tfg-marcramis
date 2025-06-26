from spade.message import Message
from spade.template import Template

from five_client.parser import MessageParser, ImageParser, DefaultParser


class MessageFactory:
    """
    Factory class to create message parsers
    """

    @staticmethod
    def create_parser(message: Message) -> MessageParser:
        if Template(metadata={"five": "image"}).match(message):
            return ImageParser()
        return DefaultParser()
