import random
from typing import Dict, List
from uuid import UUID

import networkx as nx
from aioxmpp import JID
from networkx.classes.reportviews import NodeDataView
from pyvis.network import Network


class GraphManager:

    def __init__(self) -> None:
        """
        Initialize an empty undirected graph using networkx.
        """
        self.graph: nx.Graph = nx.Graph()

    def add_agent(self, agent_jid: JID, coalition_id: None | int = None) -> None:
        """
        Add a node representing an agent with its JID, agent name (localpart) as the label,
        and optional coalition ID. The full JID is stored as an attribute.
        """
        agent_label: str = agent_jid.localpart
        if coalition_id is not None:
            self.graph.add_node(agent_label, jid=str(agent_jid.bare()), coalition_id=coalition_id)
        else:
            self.graph.add_node(agent_label, jid=str(agent_jid.bare()), coalition_id="")

    def add_connection(self, agent_jid1: JID, agent_jid2: JID) -> None:
        """
        Add an edge between two agents (neighbors) based on their JIDs.
        """
        self.graph.add_edge(agent_jid1.localpart, agent_jid2.localpart)

    def import_from_gml(self, file_path: str) -> None:
        """
        Import a graph from a GML file.
        """
        self.graph = nx.read_gml(file_path)

    def export_to_gml(self, file_path: str) -> None:
        """
        Export the current graph to a GML file.
        """
        nx.write_gml(self.graph, file_path)

    def visualize(self, output_file: str = "graph.html") -> None:
        """
        Visualize the graph using pyvis and export it as an HTML file.
        Nodes belonging to the same coalition are colored similarly.
        """
        net = Network(notebook=True)
        coalition_colors: Dict[str, str] = {}

        # Assign random colors for coalitions
        def get_coalition_color(coalition_id: str) -> str:
            if coalition_id not in coalition_colors:
                # Generate random color for the coalition
                coalition_colors[coalition_id] = f"#{random.randint(0, 0xFFFFFF):06x}"
            return coalition_colors[coalition_id]

        for node, data in self.graph.nodes(data=True):
            coalition_id = str(data.get("coalition_id"))
            color = get_coalition_color(coalition_id) if coalition_id != "" else "#808080"
            title = str(data["coalition_id"])
            if title != "":
                net.add_node(
                    node,
                    label=node,
                    title=f"Coalition: {data['coalition_id']}",
                    color=color,
                )
            else:
                net.add_node(node, label=node)

        for edge in self.graph.edges():
            net.add_edge(*edge)

        net.show(output_file)

    def get_agents_in_coalition(self, coalition_id: str, uuid: UUID | None = None) -> List[str]:
        """
        Retrieve a list of agent names (localparts) that belong to a given coalition.
        If uuid is provided, '__{uuid}' is appended to each agent's name.

        Args:
            coalition_id (str): The coalition ID to filter by.
            uuid (UUID | None): If provided, we append '__{uuid}' to each returned node name.

        Returns:
            List[str]: List of agent localparts (optionally suffixed with '__{uuid}').
        """
        nodes_with_data: NodeDataView = self.graph.nodes(data=True)
        agents = [node for node, data in nodes_with_data if data.get("coalition_id") == str(coalition_id)]
        if not uuid:
            return agents
        return [f"{agent}__{uuid}" for agent in agents]

    def list_agents(self, uuid: UUID | None = None) -> List[str]:
        """
        List all agents (nodes) in the graph by their localparts.
        If uuid is provided, '__{uuid}' is appended to each node name.
        """
        if not uuid:
            return list(self.graph.nodes)
        return [f"{n}__{uuid}" for n in self.graph.nodes]

    def list_agents_with_neighbours(self, uuid: UUID | None = None) -> Dict[str, List[str]]:
        """
        Returns a dictionary where each key is the localpart (node) and
        the value is a list of localparts for its neighbors.

        If uuid is provided, '__{uuid}' is appended to each node name
        and each neighbour name.

        Example:
            {
                "a1": ["a2", "a4"],
                "a2": ["a1", "a3"],
                ...
            }
        """
        result: Dict[str, List[str]] = {}
        for node in self.graph.nodes:
            neighbours = list(self.graph.neighbors(node))
            if uuid:
                # Append __{uuid} to both the key and the neighbors
                node_key = f"{node}__{uuid}"
                neigh_list = [f"{nbr}__{uuid}" for nbr in neighbours]
                result[node_key] = neigh_list
            else:
                # No UUID, just return raw localparts
                result[node] = neighbours
        return result

    def list_agents_jids(self, uuid: UUID | None = None) -> List[JID]:
        """
        Returns a list of JID objects from the 'jid' attribute of each node.
        If uuid is provided, '__{uuid}' is appended to each localpart name.
        """
        nodes_with_data: NodeDataView = self.graph.nodes(data=True)
        if uuid is None:
            return [JID.fromstr(data.get("jid")) for _, data in nodes_with_data]
        result: List[JID] = []
        for _, data in nodes_with_data:
            jid = JID.fromstr(data.get("jid"))
            jid_uuid = JID.fromstr(f"{jid.localpart}__{uuid}@{jid.domain}")
            result.append(jid_uuid)
        return result

    def list_agents_jids_with_neighbours(self, uuid: UUID | None = None) -> Dict[JID, List[JID]]:
        """
        Returns a dictionary where each key is the JID of a node, and
        the value is a list of JIDs corresponding to its neighbors.
        If uuid is provided, '__{uuid}' is appended to each localpart name.

        This relies on the "jid" attribute stored in each node's data.
        """
        result: Dict[JID, List[JID]] = {}
        for node, data in self.graph.nodes(data=True):
            node_jid = JID.fromstr(data["jid"])
            if uuid is not None:
                node_jid = JID.fromstr(f"{node_jid.localpart}__{uuid}@{node_jid.domain}")
            neighbour_jids: List[JID] = []
            for neighbour_node in self.graph.neighbors(node):
                neighbour_data = self.graph.nodes[neighbour_node]
                neighbour_jid = JID.fromstr(neighbour_data["jid"])
                if uuid is not None:
                    neighbour_jid = JID.fromstr(f"{neighbour_jid.localpart}__{uuid}@{neighbour_jid.domain}")
                neighbour_jids.append(neighbour_jid)
            result[node_jid] = neighbour_jids
        return result

    def list_connections(self) -> List[tuple]:
        """
        List all connections (edges) between agents.
        """
        return list(self.graph.edges)

    def generate_ring(self, agents: List[JID]) -> None:
        """
        Generate a ring structure where each agent is connected to its neighbors in a circular fashion.
        """
        self.graph = nx.Graph()
        num_agents = len(agents)
        for i in range(num_agents):
            self.add_agent(agents[i])
            self.add_connection(agents[i], agents[(i + 1) % num_agents])  # Circular connection

    def generate_complete(self, agents: List[JID]) -> None:
        """
        Generate a complete graph where each agent is connected to every other agent.
        """
        self.graph = nx.Graph()
        for agent in agents:
            self.add_agent(agent)
        for i, agent in enumerate(agents):
            for j in range(i + 1, len(agents)):
                self.add_connection(agent, agents[j])

    def generate_small_world(self, agents: List[JID], k: int = 4, p: float = 0.3) -> None:
        """
        Generate a small-world network using the Watts-Strogatz model.
        `k` is the number of nearest neighbors in the ring, and `p` is the probability of rewiring.
        """
        self.graph = nx.Graph()
        num_agents = len(agents)
        ws_graph = nx.watts_strogatz_graph(num_agents, k, p)
        for _, agent in enumerate(agents):
            self.add_agent(agent)

        for edge in ws_graph.edges():
            agent1 = agents[edge[0]]
            agent2 = agents[edge[1]]
            self.add_connection(agent1, agent2)

    def export_as_cytoscape_json(self):
        nodes = [
            {
                "data": {
                    "id": str(n),
                    "label": str(self.graph.nodes[n].get("label", "")),
                    "jid": self.graph.nodes[n].get("jid", ""),
                    "coalition_id": self.graph.nodes[n].get("coalition_id", "")
                }
            }
            for n in self.graph.nodes()
        ]

        edges = [
            {
                "data": {
                    "source": str(u),
                    "target": str(v)
                }
            }
            for u, v in self.graph.edges()
        ]

        return {"nodes": nodes, "edges": edges}