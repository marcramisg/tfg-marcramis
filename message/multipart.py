import copy
import uuid

from aioxmpp import JID
from spade.message import Message


class MultipartHandler:
    """
    Class created to handle the SPADE agents maximum message length limitation. The aioxmpp package maximum
    content length is 256 * 1024, so this class is able to split a content that exceeds a desired size into messages
    of the same desired maximum length. This class adds a header into the messages content to be able to
    rebuild the messages in the correct order. The header is "multipart#[index]/[total]#[uuid4]|" where "index"
    is the id of the current message (starting by 1), "total" is the number of messages needed to rebuild the
    original content and "uuid4" is the unique universal identifier (v4) of the original splitted message.
    """

    def __init__(self) -> None:
        # the storage is: { "ag1@localhost": { "uuid4": [ None, "msg2" ] } }
        self.__multipart_message_storage: dict[JID, dict[str, list[str | None]]] = {}
        self.__metadata_start: str = "multipart"
        self.__metadata_split_token: str = "#"
        self.__metadata_uuid: str = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"
        self.__metadata_num_messages: int = 9_999_999
        self.__metadata_end_token: str = "|"
        metadata_header = (
            self.__metadata_start
            + self.__metadata_split_token
            + f"{self.__metadata_num_messages}/{self.__metadata_num_messages}"
            + self.__metadata_split_token
            + self.__metadata_uuid
            + self.__metadata_end_token
        )
        self.__metadata_header_size: int = len(metadata_header)

    @property
    def metadata_header_size(self) -> int:
        return self.__metadata_header_size

    def is_multipart(self, message: Message) -> bool:
        return message.body.startswith(self.__metadata_start + self.__metadata_split_token)

    def get_header(self, content: str) -> str:
        return content.split(self.__metadata_end_token)[0]

    def _get_part_number(self, content: str) -> int:
        header = self.get_header(content=content)
        part_number = header.split(self.__metadata_split_token)[1].split("/")[0]
        return int(part_number)

    def _get_total_parts(self, content: str) -> int:
        header = self.get_header(content=content)
        total_parts = header.split(self.__metadata_split_token)[1].split("/")[1]
        return int(total_parts)

    def _get_uuid4(self, content: str) -> str:
        header = self.get_header(content=content)
        uuid4 = header.split(self.__metadata_split_token)[2]
        return uuid4

    def any_multipart_waiting(self) -> bool:
        return len(self.__multipart_message_storage.keys()) > 0

    def is_multipart_complete(self, message: Message) -> bool | None:
        """
        Returns a bool to denote whether the message is complete and ready to be rebuilded.
        Returns None if the sender has not multipart messages stored.

        Args:
            message (Message): A SPADE message used to check if it is part of a chain of multipart messages.

        Returns:
            bool | None: True if multipart is complete, False otherwise and None if the sender has not multipart messages stored.
        """
        sender = message.sender
        uuid4 = self._get_uuid4(message.body)
        if (
            not sender in self.__multipart_message_storage
            or not uuid4 in self.__multipart_message_storage[sender].keys()
        ):
            return None
        for part in self.__multipart_message_storage[sender][uuid4]:
            if part is None:
                return False
        return True

    def _rebuild_multipart_content(self, sender: JID, uuid4: str) -> str:
        content = ""
        for part in self.__multipart_message_storage[sender][uuid4]:
            content += part if part is not None else ""
        return content

    def __remove_data(self, sender: JID, uuid4: str) -> None:
        if sender in self.__multipart_message_storage:
            if uuid4 in self.__multipart_message_storage[sender].keys():
                del self.__multipart_message_storage[sender][uuid4]
                if len(self.__multipart_message_storage[sender].keys()) == 0:
                    del self.__multipart_message_storage[sender]

    def rebuild_multipart(self, message: Message) -> Message | None:
        """
        Rebuilds the multipart message linked to the message argument and removes the sender
        multipart stored data.

        Args:
            message (Message): One message part.

        Returns:
            Message | None: The rebuilded message with all the multiparts content in its body property.
            Returns None if the message is not completed or it is not a multipart message.
        """
        # NOTE multipart header: multipart#1/2#xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx|
        if self.is_multipart(message):
            sender = message.sender
            multipart_meta = message.body.split(self.__metadata_end_token)[0]
            part_number = self._get_part_number(message.body)
            total_parts = self._get_total_parts(message.body)
            uuid4 = self._get_uuid4(message.body)
            if not sender in self.__multipart_message_storage:
                self.__multipart_message_storage[sender] = {}
            if not uuid4 in self.__multipart_message_storage[sender].keys():
                self.__multipart_message_storage[sender][uuid4] = [None] * total_parts
            self.__multipart_message_storage[sender][uuid4][part_number - 1] = message.body[
                len(multipart_meta + self.__metadata_end_token) :
            ]
            if self.is_multipart_complete(message):
                message.body = self._rebuild_multipart_content(sender=sender, uuid4=uuid4)
                self.__remove_data(sender=sender, uuid4=uuid4)
                return message
        return None

    def __divide_content(self, content: str, size: int) -> list[str]:
        if size <= 0:
            raise RuntimeError(f"The size must be a positive integer, but the current value is: {size}")
        return [content[i : i + size] for i in range(0, len(content), size)]

    def _generate_multipart_content(self, content: str, max_size: int) -> list[str] | None:
        """
        Generates a list of multipart content based on the desired maximum size of each multipart message content
        and the maximum header size of the multipart messages metadata.

        Args:
            content (str): The content to be splitted if its length exceeds the max_size.
            max_size (int): Threshold to split the content into a list of content.

        Returns:
            list[str] | None: List of multipart message content to put in the body of the SPADE messages
            or None if the content length does not exceed the max_size tanking into account the multipart header metadata.
        """
        if max_size - self.__metadata_header_size <= 0:
            raise RuntimeError(f"The max_size message must be increased at least to {self.__metadata_header_size + 1}")

        if len(content) > max_size:
            multiparts = self.__divide_content(content, max_size - self.__metadata_header_size)
            uuid4 = str(uuid.uuid4())
            return [
                f"{self.__metadata_start}{self.__metadata_split_token}{i + 1}/{len(multiparts)}{self.__metadata_split_token}{uuid4}{self.__metadata_end_token}{part}"
                for i, part in enumerate(multiparts)
            ]
        return None

    def generate_multipart_messages(self, content: str, max_size: int, message_base: Message) -> list[Message] | None:
        """
        Creates multipart messages from one SPADE message if the length of the content
        argument is longer than the max_size argument.

        Args:
            content (str): The information that multipart messages will have in its bodies.
            max_size (int): Maximum size body length of each multipart message.
            message_base (Message): The message that will be copied by all the multipart messages, replacing just the body.

        Returns:
            list[Message] | None: A list of multipart messages to send or None if the content does not exceed the maximum size.
        """
        content_splits = self._generate_multipart_content(content=content, max_size=max_size)
        if content_splits is not None:
            multiparts_messages: list[Message] = []
            for multipart in content_splits:
                message = copy.deepcopy(message_base)
                message.body = multipart
                multiparts_messages.append(message)
            return multiparts_messages
        return None