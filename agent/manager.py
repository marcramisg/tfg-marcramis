import asyncio
import os
import subprocess
from typing import TYPE_CHECKING

from five_client.agent import AgentBase
from ..behaviour.manager import LaunchAgentsBehaviour, Wait

from aiohttp import web
from aioxmpp import JID



if TYPE_CHECKING:
    from .agent import AgentNodeBase

class AgentManager(AgentBase):
    def __init__(self,
        jid: str,
        password: str,
        agents: list["AgentNodeBase"],
        agents_coordinator: JID,
        agents_observers: list[JID],
        max_message_size: int,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.agents = agents
        self.agents_coordinator = agents_coordinator
        self.agents_observers = [] if agents_observers is None else agents_observers
        super().__init__(jid, password, max_message_size, web_address, web_port, verify_security)
        self.logger.debug(f"Agents to launch: {[a.jid.bare() for a in self.agents]}")
        self.jid_domain: str = "@" + jid.split("@")[1]

    
    async def setup(self) -> None:
        self.setup_presence_handlers()
        self.presence.set_available()
        self.add_behaviour(LaunchAgentsBehaviour())
        self.add_behaviour(Wait())

    def register_user(self, username, domain, password):
        container = "ejabberd"

        # Comprobamos si el usuario ya existe
        result = subprocess.run(
            ["docker", "exec", container, "ejabberdctl", "registered_users", domain],
            capture_output=True, text=True
        )

        if username in result.stdout:
            print(f"[INFO] Usuario {username}@{domain} ya registrado.")
            return

        # Si no está, lo registramos
        subprocess.run(
            ["docker", "exec", container, "ejabberdctl", "register", username, domain, password],
            check=True
        )
        print(f"[INFO] Usuario {username}@{domain} registrado correctamente.")

        
    async def launch_agents(self) -> None:
        """
        Starts the FL agents.
        """
        while not self.agents:  # Esperar hasta que haya agentes
            self.logger.warning("Waiting for agents to be created...")
            await asyncio.sleep(1)
        self.logger.debug(f"Initializating launch of {[str(a.jid.bare()) for a in self.agents]}")
        for agent in self.agents:
            neighbour_jids = agent.neighbours
            self.logger.debug(
                f"The neighbour JIDs for agent {agent.jid.bare()} are {[str(j.bare()) for j in neighbour_jids]}"
            )

        for agent in self.agents:
            self.register_user(str(agent.jid.localpart), str("localhost"), str(agent.password))
            await agent.start(auto_register=True)
            


    async def stop_agents(self):   
        for agent in self.agents:
            await agent.stop()

        while any(agent.is_alive() for agent in self.agents):
            await asyncio.sleep(1)

        self.agents.clear() 
        print("Agents finished.")

    async def pause_agent(self, jid):   
        agent_to_remove = None  
        for agent in self.agents:  
            if str(agent.jid).split("@")[0].split("_")[0] == jid:  
                agent_to_remove = agent  
                break  

        if agent_to_remove:  

            agent_to_remove.presence.set_unavailable()  
            print(f"Agent {jid} stopped.")  
        else:  
            print(f"Agent {jid} not found.")

    async def resume_agent(self, jid):   
        agent_to_resume = None  
        for agent in self.agents:  
            if str(agent.jid).split("@")[0].split("_")[0] == jid:  
                agent_to_resume = agent  
                break  

        if agent_to_resume:  
            agent_to_resume.presence.set_available()
            agent_to_resume.current_round = agent_to_resume.global_round  
            print(f"Agent {jid} resumed.")  
        else:  
            print(f"Agent {jid} not found.")

    async def stop(self):
        await self.stop_agents()
        await super().stop()

    async def update_neighbours(self, source, target, action):

        agent_source = next(
            (agent for agent in self.agents if str(agent.jid).startswith(f"{source.lower()}__")),
            None
        )
        agent_target = next(
            (agent for agent in self.agents if str(agent.jid).startswith(f"{target.lower()}__")),
            None
        )

        if action == "create_edge":
            agent_source.neighbours.append(agent_target.jid)
            agent_target.neighbours.append(agent_source.jid)
            print(f"Creating link between {agent_source.jid} and {agent_target.jid}")
        elif action == "delete_edge":
            agent_source.neighbours.remove(agent_target.jid)
            agent_target.neighbours.remove(agent_source.jid)
            print(f"Removing link between {agent_source.jid} and {agent_target.jid}")
        
        return [agent_source,agent_target]