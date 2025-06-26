import asyncio
import traceback
from typing import TYPE_CHECKING

from aioxmpp import JID
from spade.behaviour import FSMBehaviour, State
from spade.message import Message
from spade.presence import PresenceManager

if TYPE_CHECKING:
    from ..agent.agent import AgentNodeBase
    from ..agent.coordinator import CoordinatorAgent

# --------------------------------------------- #
# --------------------------------------------- #
# --------------------------------------------- #


class AvailableNodeState(State):

    def __init__(self, coordinator: JID):
        self.coordinator = coordinator
        self.loops: int = 0
        super().__init__()

    async def on_start(self) -> None:
        if self.loops < 1:
            agent: "AgentNodeBase" = self.agent
            agent.presence.set_available()
            agent.logger.debug("Available.")
            try:

                coordinator = str(self.coordinator.bare())
                message = Message(to=coordinator, sender=str(agent.jid.bare()))
            except Exception as e:
                traceback.print_exc()
            message.body = "ready to subscribe"
            message.set_metadata("rf.presence", "sync")
            await self.send(message)
            agent.logger.debug(f"'{message.body}' message sent to {coordinator}.")

    async def run(self) -> None:
        agent: "AgentNodeBase" = self.agent
        self.set_next_state("available")
        msg = await self.receive(timeout=1)
        if msg and msg.body == "start to subscribe":
            try:
                jid = msg.sender.bare()
            except Exception as e:
                traceback.print_exc()
            agent.logger.debug(f"'{msg.body}' message received from {jid}.")
            self.set_next_state("subscription")

    async def on_end(self) -> None:
        self.loops += 1


class SubscriptionNodeState(State):

    def __init__(self, coordinator: JID):
        self.agent: "AgentNodeBase"
        self.coordinator = coordinator
        self.loops: int = 0
        self.ready_to_start = False
        super().__init__()

    async def on_start(self) -> None:
        if self.loops < 1:
            try:
                agent: "AgentNodeBase" = self.agent
                agent.logger.debug("Subscribing to neighbours...")
                agent.subscribe_to_neighbours()
            except Exception:
                traceback.print_exc()

    async def run(self) -> None:
        try:
            agent: "AgentNodeBase" = self.agent
            if not self.ready_to_start and agent.is_presence_completed():
                self.ready_to_start = True
                try:
                    coordinator = str(self.coordinator.bare())
                    message = Message(to=coordinator, sender=str(agent.jid.bare()))
                except Exception as e:
                    traceback.print_exc()
                message.body = "ready to start"
                message.set_metadata("rf.presence", "sync")
                await self.send(message)
                agent.logger.info(f"Available neighbours: {[n.localpart for n in agent.get_available_neighbours()]}")
                agent.logger.debug(f"'{message.body}' message sent to {coordinator}")
            elif not agent.is_presence_completed():
                contacts: dict[JID, dict] = agent.presence.get_contacts()
                try:
                    subscription_status = [(str(jid.bare()), data["subscription"]) for jid, data in contacts.items()]
                except Exception as e:
                    traceback.print_exc()
                agent.logger.debug(f"Neighbour's subscription status is {subscription_status}")
                for jid, status in agent.get_non_subscribe_both_neighbours().items():
                    try:
                        j = str(jid.bare())
                    except Exception as e:
                        traceback.print_exc()
                    agent.presence.subscribe(j)
                    agent.logger.debug(f"Sent subscription request to {j} because status is '{status}'.")
                await asyncio.sleep(5)

            msg = await self.receive(timeout=1)
            if msg and msg.body == "start the algorithm":
                try:
                    jid = msg.sender.bare()
                except Exception as e:
                    traceback.print_exc()
                agent.logger.debug(f"'{msg.body}' message received from {jid}.")
                for behaviour, template in agent.post_coordination_behaviours:
                    agent.add_behaviour(behaviour, template)
                    agent.logger.debug(f"Behaviour {type(behaviour)} added with template {template}.")
                agent.logger.info("Coordination phase ended successfully.")
                self.kill()
            else:
                self.set_next_state("subscription")
        except Exception:
            traceback.print_exc()

    async def on_end(self) -> None:
        self.loops += 1


class PresenceNodeFSM(FSMBehaviour):

    def __init__(self, coordinator: JID):
        self.agent: "AgentNodeBase"
        self.coordinator = coordinator
        super().__init__()

    def setup(self) -> None:
        self.add_state(name="available", state=AvailableNodeState(self.coordinator), initial=True)
        self.add_state(name="subscription", state=SubscriptionNodeState(self.coordinator))
        self.add_transition(source="available", dest="available")
        self.add_transition(source="available", dest="subscription")
        self.add_transition(source="subscription", dest="subscription")

    async def on_end(self) -> None:
        self.agent.logger.debug("PresenceSetupFSM behaviour finished.")


# --------------------------------------------- #
# --------------------------------------------- #
# --------------------------------------------- #


class AvailableCoordinatorState(State):

    def __init__(self, coordinated_agents: list[JID]):
        self.loops: int = 0
        self.ready_agents: dict[str, bool] = {str(jid.bare()): False for jid in coordinated_agents}
        super().__init__()

    async def on_start(self) -> None:
        if self.loops < 1:
            agent: "CoordinatorAgent" = self.agent
            agent.logger.debug(f"AvailableCoordinatorState: {self.ready_agents}.")

    async def run(self) -> None:
        agent: "CoordinatorAgent" = self.agent
        self.set_next_state("available")
        if not self._are_all_agents_ready():
            msg = await self.receive(timeout=3)
            if msg and msg.body == "ready to subscribe":
                try:
                    jid = msg.sender.bare()
                except Exception as e:
                    traceback.print_exc()
                self.ready_agents[str(jid)] = True
                agent.logger.debug(f"AvailableCoordinatorState: {self.ready_agents}.")
                agent.logger.debug(f"'{msg.body}' message received from {jid}.")

        if self._are_all_agents_ready():
            body = "start to subscribe"
            for jid in self.ready_agents.keys():
                try:
                    msg = Message(to=jid, sender=str(agent.jid.bare()))
                except Exception as e:
                    traceback.print_exc()
                msg.body = body
                msg.set_metadata("rf.presence", "sync")
                await self.send(msg)

            agent.logger.info(f"All '{body}' messages sent.")
            self.set_next_state("subscription")

    async def on_end(self) -> None:
        self.loops += 1

    def _are_all_agents_ready(self) -> bool:
        return all(self.ready_agents.values())


class SubscriptionCoordinatorState(State):

    def __init__(self, coordinated_agents: list[JID]):
        self.loops: int = 0
        self.ready_agents: dict[str, bool] = {str(jid.bare()): False for jid in coordinated_agents}
        super().__init__()

    async def on_start(self) -> None:
        if self.loops < 1:
            agent: "CoordinatorAgent" = self.agent
            agent.logger.debug(f"SubscriptionCoordinatorState: {self.ready_agents}.")

    async def run(self) -> None:
        agent: "CoordinatorAgent" = self.agent
        if not self._are_all_agents_ready():
            msg = await self.receive(timeout=3)
            if msg and msg.body == "ready to start":
                try:
                    jid = msg.sender.bare()
                except Exception as e:
                    traceback.print_exc()
                self.ready_agents[str(jid)] = True
                agent.logger.debug(f"SubscriptionCoordinatorState: {self.ready_agents}.")
                agent.logger.debug(f"'{msg.body}' message received from {jid}.")

        if self._are_all_agents_ready():
            body = "start the algorithm"
            for jid in self.ready_agents.keys():
                try:
                    msg = Message(to=jid, sender=str(agent.jid.bare()))
                except Exception as e:
                    traceback.print_exc()
                msg.body = body
                msg.set_metadata("rf.presence", "sync")
                await self.send(msg)
            agent.logger.info(f"All '{body}' messages sent.")
            agent.ready_to_start_algorithm = True
            self.set_next_state("wait")
        else:
            self.set_next_state("subscription")

    async def on_end(self) -> None:
        self.loops += 1

    def _are_all_agents_ready(self) -> bool:
        return all(self.ready_agents.values())


class WaitState(State):

    async def run(self) -> None:
        await asyncio.sleep(10)


class PresenceCoordinatorFSM(FSMBehaviour):

    def __init__(
        self,
        coordinated_agents: list[JID],
    ):
        self.coordinated_agents = coordinated_agents
        super().__init__()

    async def on_start(self) -> None:
        agent: "CoordinatorAgent" = self.agent
        agent.logger.debug("PresenceCoordinatorFSM started.")

    def setup(self) -> None:
        self.add_state(
            name="available",
            state=AvailableCoordinatorState(self.coordinated_agents),
            initial=True,
        )
        self.add_state(
            name="subscription",
            state=SubscriptionCoordinatorState(self.coordinated_agents),
        )
        self.add_state(name="wait", state=WaitState())
        self.add_transition(source="available", dest="available")
        self.add_transition(source="available", dest="subscription")
        self.add_transition(source="subscription", dest="subscription")
        self.add_transition(source="subscription", dest="wait")

    async def on_end(self) -> None:
        agent: "CoordinatorAgent" = self.agent
        agent.logger.debug("PresenceCoordinatorFSM behaviour finished.")
