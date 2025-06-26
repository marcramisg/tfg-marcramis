import copy
import json
from datetime import datetime, timezone
from typing import Any, OrderedDict

from aioxmpp import JID
from spade.message import Message


class SimilarityVector:
    def __init__(
        self,
        vector: OrderedDict[str, float],  # str is the name of the layer and float is the similarity coefficient
        owner: None | JID = None,
        request_reply: None | bool = False,
        algorithm_iteration: None | int = None,
        sent_time_z: None | datetime = None,
        received_time_z: None | datetime = None,
    ):
        self.vector = vector
        self.owner = owner
        self.request_reply: bool = request_reply if request_reply is not None else False
        self.algorithm_iteration = algorithm_iteration
        self.sent_time_z = sent_time_z
        self.received_time_z = received_time_z

    def to_message(self, message: None | Message = None) -> Message:
        msg = Message() if message is None else copy.deepcopy(message)
        content: dict[str, Any] = {}
        content["vector"] = self.vector
        content["owner"] = str(self.owner.bare()) if self.owner else None
        content["request_reply"] = self.request_reply
        content["algorithm_iteration"] = self.algorithm_iteration
        sent_time_z = datetime.now(tz=timezone.utc) if self.sent_time_z is None else self.sent_time_z
        content["sent_time_z"] = sent_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        msg.body = json.dumps(content)
        return msg

    @staticmethod
    def from_message(message: Message) -> "SimilarityVector":
        content: dict[str, Any] = json.loads(message.body)
        vector: OrderedDict[str, float] = content["vector"]
        request_reply: bool = content["request_reply"]
        algorithm_iteration: int | None = content["algorithm_iteration"] if "algorithm_iteration" in content else None
        sent_time_z: datetime = datetime.strptime(content["sent_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        received_time_z: datetime = datetime.now(tz=timezone.utc)
        if "received_time_z" in content:
            received_time_z = datetime.strptime(content["received_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc
            )
        return SimilarityVector(
            vector=vector,
            owner=message.sender,
            request_reply=request_reply,
            algorithm_iteration=algorithm_iteration,
            sent_time_z=sent_time_z,
            received_time_z=received_time_z,
        )
