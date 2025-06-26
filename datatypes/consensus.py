import copy
import json
from datetime import datetime, timezone
from typing import Any, OrderedDict

from aioxmpp import JID
from spade.message import Message
from torch import Tensor

from .models import ModelManager


class Consensus:
    """
    Stores consensus information during layer transmission and processing.
    """

    def __init__(
        self,
        layers: OrderedDict[str, Tensor],
        sender: None | JID = None,
        request_reply: None | bool = None,
        sent_time_z: None | datetime = None,
        received_time_z: None | datetime = None,
        processed_start_time_z: None | datetime = None,
        processed_end_time_z: None | datetime = None,
    ):
        self.layers = layers
        self.sender = sender
        self.request_reply = request_reply if request_reply is not None else False
        self.sent_time_z = sent_time_z
        self.received_time_z = received_time_z
        self.processed_start_time_z = processed_start_time_z
        self.processed_end_time_z = processed_end_time_z

        self.__check_utc(self.sent_time_z)
        self.__check_utc(self.received_time_z)
        self.__check_utc(self.processed_start_time_z)
        self.__check_utc(self.processed_end_time_z)

    def to_message(self, message: None | Message = None) -> Message:
        msg = Message() if message is None else copy.deepcopy(message)
        content: dict[str, Any] = {}
        base64_layers = ModelManager.export_layers(self.layers)
        content["layers"] = base64_layers
        content["sender"] = str(self.sender.bare()) if self.sender is not None else None
        content["request_reply"] = self.request_reply
        sent_time_z = datetime.now(tz=timezone.utc) if self.sent_time_z is None else self.sent_time_z
        content["sent_time_z"] = sent_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        msg.body = json.dumps(content)
        return msg

    @staticmethod
    def from_message(message: Message) -> "Consensus":
        content: dict[str, Any] = json.loads(message.body)
        base64_layers: str = content["layers"]
        request_reply: bool = bool(content["request_reply"])
        layers = ModelManager.import_layers(base64_layers)
        sent_time_z: datetime = datetime.strptime(content["sent_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        received_time_z: datetime = datetime.now(tz=timezone.utc)
        if "received_time_z" in content:
            received_time_z = datetime.strptime(content["received_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc
            )
        processed_start_time_z: None | datetime = None
        if "processed_start_time_z" in content:
            processed_start_time_z = datetime.strptime(
                content["processed_start_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=timezone.utc)
        processed_end_time_z: None | datetime = None
        if "processed_end_time_z" in content:
            processed_end_time_z = datetime.strptime(content["processed_end_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc
            )
        return Consensus(
            layers=layers,
            sender=message.sender,
            request_reply=request_reply,
            sent_time_z=sent_time_z,
            received_time_z=received_time_z,
            processed_start_time_z=processed_start_time_z,
            processed_end_time_z=processed_end_time_z,
        )

    def __str__(self) -> str:
        content: dict[str, Any] = {}
        base64_layers = ModelManager.export_layers(self.layers)
        content["layers"] = base64_layers
        content["request_reply"] = self.request_reply
        content["sender"] = self.sender
        if self.sent_time_z is not None:
            content["sent_time_z"] = self.sent_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if self.received_time_z is not None:
            content["received_time_z"] = self.received_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if self.processed_start_time_z is not None:
            content["processed_start_time_z"] = self.processed_start_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if self.processed_end_time_z is not None:
            content["processed_end_time_z"] = self.processed_end_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return json.dumps(content)

    def __check_utc(self, dt: None | datetime) -> None:
        if dt is not None:
            if dt.tzinfo is None or dt.tzinfo != timezone.utc:
                raise ValueError("All Consensus datetimes must be timezone-aware (UTC) (Z).")
