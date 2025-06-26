from spade.behaviour import FSMBehaviour
from spade.behaviour import CyclicBehaviour
from aioxmpp import JID, PresenceType
# from five_client.agent import AgentBase


class FiveBehaviour(FSMBehaviour):
    async def on_start(self) -> None:
        print(f"[{self.agent.name}] Agent started FSM.")

    def on_available(self, jid, stanza) -> None:
        print(f"[{self.agent.name}] Agent {jid} is available.")

    def on_subscribed(self, jid) -> None:
        print(f"[{self.agent.name}] Agent {jid} has accepted the subscription.")
        print(f"[{self.agent.name}] Contacts List: {self.presence.get_contacts()}")

    def on_subscribe(self, jid) -> None:
        print(
            f"[{self.agent.name}] Agent {jid} asked for subscription. Let's aprove it."
        )
        self.presence.approve(jid)

    async def run(self):
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available

        self.presence.set_available()

    async def on_end(self):
        self.presence.set_unavailable()
        await self.agent.stop()



# class PerceptionState(State):
#     def __init__(self, shell: AgentShell):
#         super().__init__()
#         self.__shell = shell

#     def on_available(self, jid, stanza) -> None:
#         print(f"[{self.agent.name}] Agent {jid} is available.")

#     def on_subscribed(self, jid) -> None:
#         print(f"[{self.agent.name}] Agent {jid} has accepted the subscription.")
#         print(f"[{self.agent.name}] Contacts List: {self.presence.get_contacts()}")

#     def on_subscribe(self, jid) -> None:
#         print(
#             f"[{self.agent.name}] Agent {jid} asked for subscription. Let's aprove it."
#         )
#         self.presence.approve(jid)

#     async def run(self):
#         self.presence.on_subscribe = self.on_subscribe
#         self.presence.on_subscribed = self.on_subscribed
#         self.presence.on_available = self.on_available

#         self.presence.set_available()
