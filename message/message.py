from typing import Dict

from spade.message import Message


class RfMessage(Message):

    def __init__(
        self,
        to: str | None = None,
        sender: str | None = None,
        body: str | None = None,
        thread: str | None = None,
        metadata: Dict[str, str] | None = None,
        is_multipart: bool = False,
        is_multipart_completed: bool = False,
    ) -> None:
        super().__init__(to, sender, body, thread, metadata)
        self.is_multipart = is_multipart
        self.is_multipart_completed = is_multipart_completed

    def to_message(self) -> Message:
        to = None if not self.to else str(self.to.bare())
        sender = None if not self.sender else str(self.sender.bare())
        return Message(
            to=to,
            sender=sender,
            body=self.body,
            thread=self.thread,
            metadata=self.metadata,
        )

    @staticmethod
    def from_message(message: Message, is_multipart: bool, is_multipart_completed: bool) -> "RfMessage":
        to = None if not message.to else str(message.to.bare())
        sender = None if not message.sender else str(message.sender.bare())
        return RfMessage(
            to=to,
            sender=sender,
            body=message.body,
            thread=message.thread,
            metadata=message.metadata,
            is_multipart=is_multipart,
            is_multipart_completed=is_multipart_completed,
        )

    @staticmethod
    def is_completed(message: Message | None) -> bool:
        if message is None:
            return False
        if isinstance(message, RfMessage):
            return not message.is_multipart or (message.is_multipart and message.is_multipart_completed)
        return True

    @staticmethod
    def is_multipart_and_not_yet_completed(message: Message | None) -> bool:
        if isinstance(message, RfMessage):
            return message.is_multipart and not message.is_multipart_completed
        return False