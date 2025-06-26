import traceback
from abc import ABCMeta, abstractmethod
from queue import Queue
from typing import TYPE_CHECKING, Optional, OrderedDict

from aioxmpp import JID, PresenceType
from aioxmpp.stanza import Presence
from spade.agent import Agent
from spade.message import Message
from spade.template import Template
from torch import Tensor


import importlib.util
import sys
from pathlib import Path
import traceback
import time
import asyncio


from five_client.behaviour import ObserverBehaviour
from five_client.data import DigitalPositionData
from five_client.data.camera import CameraData

from ..behaviour.coordination import PresenceNodeFSM
from ..behaviour.premiofl.fsm import PremioFsmBehaviour
from ..behaviour.premiofl.layer_receiver import LayerReceiverBehaviour
from ..behaviour.premiofl.similarity_receiver import SimilarityReceiverBehaviour
from ..message.message import RfMessage
from ..message.multipart import MultipartHandler
from ..datatypes.consensus import Consensus
from ..datatypes.consensus_manager import ConsensusManager
from ..datatypes.models import ModelManager
from ..log.general import GeneralLogManager
from ..log.message import MessageLogManager
from ..log.algorithm import AlgorithmLogManager
from ..log.general import GeneralLogManager
from ..log.message import MessageLogManager
from ..log.nn import NnConvergenceLogManager, NnInferenceLogManager, NnTrainLogManager
from ..similarity.similarity_manager import SimilarityManager
from ..similarity.similarity_vector import SimilarityVector

if TYPE_CHECKING:
    from spade.behaviour import CyclicBehaviour

class AgentBase(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int = 250_000, # int,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        extra_log_name = f"agent.{JID.fromstr(jid).localpart}"
        self.logger = GeneralLogManager(extra_logger_name=extra_log_name)
        self.message_logger = MessageLogManager(extra_logger_name=extra_log_name)
        self.max_message_size = max_message_size
        self.web_address = web_address
        self.web_port = web_port
        self._multipart_handler = MultipartHandler()
        super().__init__(jid=jid, password=password, verify_security=verify_security)

    async def setup(self) -> None:
        self.setup_presence_handlers()

    async def send(self, message: Message, behaviour: Optional["CyclicBehaviour"] = None) -> None:
        messages = self._multipart_handler.generate_multipart_messages(
            content=message.body,
            max_size=self.max_message_size,
            message_base=message,
        )
        if messages is None:
            messages = [message]
        for msg in messages:
            if behaviour is not None:
                await behaviour.send(msg=msg)
            else:
                futures = self.dispatch(msg=msg)
                for f in futures:
                    f.result()
            self.logger.debug(f"Message ({msg.sender.bare()}) -> ({msg.to.bare()}): {msg.body}")

    async def receive(self, behaviour: "CyclicBehaviour", timeout: None | float = 0) -> RfMessage | None:
        """
        Put a behaviour start listening to messages using the MultipartHandler class.
        If a message arrives, this function returns the message, otherwise returns None.
        If the message is a multipart message and it is completed, returns the completed
        message with the flags `is_multipart` and `is_multipart_completed` set to True.

        Args:
            behaviour (CyclicBehaviour): The receiver behaviour.
            timeout (None | float, optional): Timeout in seconds. Defaults to 0.

        Returns:
            RfMessage | None: The message received -and rebuilded if necessary- or None
            if not messages received after timeout ends.
        """
        msg: Message | None = await behaviour.receive(timeout=timeout)
        if msg is not None:
            is_multipart = self._multipart_handler.is_multipart(msg)
            if is_multipart:
                header = self._multipart_handler.get_header(msg.body)
                self.logger.debug(f"Multipart message received from {msg.sender}: {header} with length {len(msg.body)}")
                multipart_msg = self._multipart_handler.rebuild_multipart(message=msg)
                is_multipart_completed = multipart_msg is not None
                if is_multipart_completed:
                    return RfMessage.from_message(
                        message=multipart_msg,
                        is_multipart=is_multipart,
                        is_multipart_completed=is_multipart_completed,
                    )
                return RfMessage.from_message(
                    message=msg,
                    is_multipart=is_multipart,
                    is_multipart_completed=is_multipart_completed,
                )
            self.logger.debug(f"Message received from {msg.sender}: with length {len(msg.body)}")
            return RfMessage.from_message(message=msg, is_multipart=False, is_multipart_completed=False)
        return None

    def any_multipart_waiting(self) -> bool:
        return self._multipart_handler.any_multipart_waiting()

    def on_available(self, jid: str, stanza) -> None:
        self.logger.debug(f"{jid} is available with stanza {stanza}.")

    def on_subscribed(self, jid) -> None:
        self.logger.debug(f"{jid} has accepted my subscription request.")
        self.logger.debug(f"My contact list is {self.presence.get_contacts()}.")

    def on_subscribe(self, jid) -> None:
        self.presence.approve(jid)
        self.logger.debug(f"{jid} approved.")

    def setup_presence_handlers(self) -> None:
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available

class AgentNodeBase(AgentBase):
    def __init__(
       self,
        jid: str,
        password: str,
        max_message_size: int,
        observers: None | list[JID] = None,
        neighbours: None | list[JID] = None,
        coordinator: None | JID = None,
        post_coordination_behaviours: None | list[tuple["CyclicBehaviour", Template]] = None,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
        global_round: int = 0,
    ):
        self.observers = [] if observers is None else observers
        self.neighbours = [] if neighbours is None else neighbours
        self.coordinator = coordinator
        self.post_coordination_behaviours = [] if post_coordination_behaviours is None else post_coordination_behaviours
        self.coordination_fsm: "PresenceNodeFSM" | None = None
        super().__init__(
            jid=jid,
            password=password,
            max_message_size=max_message_size,
            web_address=web_address,
            web_port=web_port,
            verify_security=verify_security,
        )

    async def setup(self) -> None:

        await super().setup()
        if self.coordinator is not None:
            self.coordination_fsm = PresenceNodeFSM(self.coordinator)
            template = Template()
            template.set_metadata("rf.presence", "sync")
            self.add_behaviour(self.coordination_fsm, template)
            self.logger.info("PresenceNodeFSM attached.")
        else:
            self.logger.info("Starting without PresenceNodeFSM.")
            for behaviour, template in self.post_coordination_behaviours:
                self.add_behaviour(behaviour, template)

    def is_starter_position_a_spawner(self) -> bool:
        """
        Returns True if the agent is spawned in a Spawner.

            bool: True if the agent is spawned using a Spawner tag name.
        Returns:
        """
        return isinstance(self.starter_position, str)
    

    
    def subscribe_to_neighbours(self) -> None:
        try:
            for jid in self.neighbours:
                self.presence.subscribe(str(jid.bare()))
                self.logger.debug(f"Subscription request sent to {jid}")
        except Exception:
            traceback.print_exc()

    def get_non_subscribe_both_neighbours(self) -> dict[JID, str]:
        contacts: dict[JID, dict[str, str | Presence]] = self.presence.get_contacts()
        result = {
            j.bare(): data["subscription"]
            for j, data in contacts.items()
            if j in self.neighbours and data["subscription"] != "both"
        }
        for jid in self.neighbours:
            if jid.bare() not in result.keys():
                result[jid.bare()] = "null"
        return result

    def is_presence_completed(self) -> bool:
        contacts: dict[JID, dict[str, str | Presence]] = self.presence.get_contacts()
        if not all(ag.bare() in contacts for ag in self.neighbours):
            return False
        return all(data["subscription"] == "both" for data in contacts.values())

    def get_available_neighbours(self) -> list[JID]:
        available_contacts: list[JID] = []
        contacts: dict[JID, dict[str, str | Presence]] = self.presence.get_contacts()
        # contacts example: a1@localhost: {'subscription': 'both', 'ask': 'subscribe',
        # 'presence': <presence from='a1@localhost/lnvf1R8J' to='a0@localhost' id=':sw_LBqbBrY8pBucyD023'
        # type=<PresenceType.AVAILABLE: None>>}
        for agent, contact_info in contacts.items():
            if "presence" in contact_info:
                presence: Presence = contact_info["presence"]
                if presence.type_ == PresenceType.AVAILABLE:
                    available_contacts.append(agent.bare())
        return available_contacts
    
    async def sync_neighbours_subscription(self) -> None:
        try:
            non_both_neighbours = self.get_non_subscribe_both_neighbours()
            
            for jid, status in non_both_neighbours.items():
                self.presence.subscribe(str(jid))
                self.logger.debug(f"Subscription request sent to {jid} (status was '{status}')")
            
            contacts = self.presence.get_contacts()
            current_neighbours_bare = {jid.bare() for jid in self.neighbours}
            
            for contact_jid in contacts.keys():
                if contact_jid.bare() not in current_neighbours_bare:
                    self.presence.unsubscribe(str(contact_jid.bare()))
                    self.logger.debug(f"Unsubscribed from ex-neighbour: {contact_jid.bare()}")
                    
            await asyncio.sleep(1.5)
            if self.is_presence_completed():
                self.logger.debug("Subscription sync completed successfully")
                return True
            else:
                self.logger.warning("Subscription sync incomplete")
                return False
        except Exception:
            traceback.print_exc()
            return False



class CoalitionAgentNodeBase(AgentNodeBase):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        coalition_id: int,
        observers: None | list[JID] = None,
        neighbours: None | list[JID] = None,
        coalitions: None | dict[int, JID] = None,
        coordinator: None | JID = None,
        post_coordination_behaviours: None | list[tuple["CyclicBehaviour", Template]] = None,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ) -> None:
        self.coalition_id = coalition_id
        neighbours = [] if neighbours is None else neighbours
        self.coalitions = {} if coalitions is None else coalitions
        coalition_neighbours = self.coalitions.values()
        if len(coalition_neighbours) != len(neighbours):
            raise ValueError(f"Coalition dict of agent {jid} must have all the neighbours information")
        for neighbour in self.neighbours:
            if neighbour not in coalition_neighbours:
                raise ValueError(
                    f"Coalition dict of agent {jid} must have all the neighbours information,"
                    + f" but {neighbour} is not in coalitions: {self.coalitions}."
                )

        super().__init__(
            jid,
            password,
            max_message_size,
            observers,
            neighbours,
            coordinator,
            post_coordination_behaviours,
            web_address,
            web_port,
            verify_security,
        )

    def get_coalition_neighbours(self) -> list[JID]:
        return [] if self.coalition_id not in self.coalitions else self.coalitions[self.coalition_id]
    


class PremioFlAgent(AgentNodeBase, metaclass=ABCMeta):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        consensus_manager: ConsensusManager,
        model_manager: ModelManager,
        similarity_manager: SimilarityManager,
        observers: list[JID] | None = None,
        neighbours: list[JID] | None = None,
        coordinator: JID | None = None,
        max_rounds: None | int = 100,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        extra_name = f"agent.{JID.fromstr(jid).localpart}"
        self.consensus_manager = consensus_manager
        self.model_manager = model_manager
        self.similarity_manager = similarity_manager
        self.max_rounds = max_rounds  # None = inf
        self.current_round: int = 0
        self.consensus_transmissions: Queue[Consensus] = Queue()
        self.message_logger = MessageLogManager(extra_logger_name=extra_name)
        self.algorithm_logger = AlgorithmLogManager(extra_logger_name=extra_name)
        self.nn_train_logger = NnTrainLogManager(extra_logger_name=extra_name)
        self.nn_inference_logger = NnInferenceLogManager(extra_logger_name=extra_name)
        self.nn_convergence_logger = NnConvergenceLogManager(extra_logger_name=extra_name)

        self.fsm_behaviour = PremioFsmBehaviour()
        self.layer_receiver_behaviour = LayerReceiverBehaviour()
        self.similarity_receiver_behaviour = SimilarityReceiverBehaviour()
        post_coordination_behaviours = [
            (self.fsm_behaviour, None),
            (
                self.layer_receiver_behaviour,
                Template(metadata={"rf.conversation": "layers"}),
            ),
            (
                self.similarity_receiver_behaviour,
                Template(metadata={"rf.conversation": "similarity"}),
            ),
        ]

        super().__init__(
            jid,
            password,
            max_message_size,
            observers,
            neighbours,
            coordinator,
            post_coordination_behaviours,
            web_address,
            web_port,
            verify_security,
        )

    def select_neighbours(self) -> list[JID]:
        """
        Get the selected available neighbours to share the model layers, based on the implementation criteria.

        Raises:
            NotImplementedError: _select_neighbours must be overrided or it raises this error.

        Returns:
            list[JID]: The list of the selected available neighbours.
        """
        return self._select_neighbours(self.get_available_neighbours())

    @abstractmethod
    def _select_neighbours(self, neighbours: list[JID]) -> list[JID]:
        raise NotImplementedError

    @abstractmethod
    def _assign_layers(
        self,
        my_vector: None | SimilarityVector,
        neighbours_vectors: dict[JID, SimilarityVector],
        selected_neighbours: list[JID],
    ) -> dict[JID, OrderedDict[str, Tensor]]:
        """
        Assigns which layers will be sent to each neighbour. In the paper this function is coined as `S_L_N`.

        Args:
            my_vector (SimilarityVector): The neighbours that will receive the layers of the neural network model.
            neighbours_vectors (dict[JID, SimilarityVector]): All neighbours' vectors, here are selected and non-selected neighbours.
            selected_neighbours (list[JID]): The neighbours that will receive the layers of the neural network model.

        Raises:
            NotImplementedError: This function must be overrided or it raises this error.

        Returns:
            dict[JID, OrderedDict[str, Tensor]]: The keys are the neighbour's `aioxmpp.JID`s and the values are the
            layer names with the `torch.Tensor` weights or biases.
        """
        raise NotImplementedError

    def assign_layers(
        self,
        selected_neighbours: list[JID],
    ) -> dict[JID, OrderedDict[str, Tensor]]:
        """
        Assigns which layers will be sent to each neighbour. In the paper this function is coined as `S_L_N`.

        Args:
            selected_neighbours (list[JID]): The neighbours that will receive the layers of the neural network model.

        Raises:
            NotImplementedError: This function must be overrided or it raises this error.

        Returns:
            dict[JID, OrderedDict[str, Tensor]]: The keys are the neighbour's `aioxmpp.JID`s and the values are the
            layer names with the `torch.Tensor` weights or biases.
        """
        return self._assign_layers(
            my_vector=self.similarity_manager.get_own_similarity_vector(),
            neighbours_vectors=self.similarity_manager.similarity_vectors,
            selected_neighbours=selected_neighbours,
        )

    async def send_similarity_vector(
        self,
        neighbour: JID,
        vector: SimilarityVector,
        thread: None | str = None,
        metadata: None | dict[str, str] = None,
        behaviour: Optional["CyclicBehaviour"] = None,
    ) -> None:
        msg = vector.to_message()
        msg.sender = str(self.jid.bare())
        msg.to = str(neighbour.bare())
        msg.thread = thread
        msg.metadata = metadata
        tag = "-REQREPLY" if vector.request_reply else ""
        await self.__send_message(message=msg, behaviour=behaviour, log_tag=f"-SIMILARITY{tag}")

    async def send_local_layers(
        self,
        neighbour: JID,
        request_reply: bool,
        layers: OrderedDict[str, Tensor],
        thread: None | str = None,
        metadata: None | dict[str, str] = None,
        behaviour: Optional["CyclicBehaviour"] = None,
    ) -> None:
        ct = Consensus(layers=layers, sender=self.jid, request_reply=request_reply)
        msg = ct.to_message()
        msg.sender = str(self.jid.bare())
        msg.to = str(neighbour.bare())
        msg.thread = thread
        msg.metadata = metadata
        tag = "-REQREPLY" if request_reply else ""
        await self.__send_message(message=msg, behaviour=behaviour, log_tag=f"-LAYERS{tag}")

    async def __send_message(self, message: Message, behaviour: "CyclicBehaviour", log_tag: str = "") -> None:
        await self.send(message=message, behaviour=behaviour)
        self.message_logger.log(
            current_round=self.current_round,
            sender=message.sender,
            to=message.to,
            msg_type=f"SEND{log_tag}",
            size=len(message.body),
            thread=message.thread,
        )

    def are_max_iterations_reached(self) -> bool:
        return self.max_rounds is not None and self.current_round > self.max_rounds

    async def stop(self) -> None:
        await super().stop()
        self.logger.info("Agent stopped.")
