from typing import Union
from aioxmpp import JID

from pathlib import Path

from five_client.data import DigitalPositionData
from five_client.data.camera import CameraData


# Realmente no tiene sentido porque diferentes agentes => diferente info requerida
# Borrar en un futuro
class AgentData:
    def __init__(
        self,
        jid: Union[str, JID],
        password: str,
        enable_agent_collision: bool,
        prefab_name: str,
        starter_position: Union[DigitalPositionData, str],
        behaviour_path: Union[str, Path],
        server_jid: Union[str, JID],
        manager_jid: Union[str, JID],
        neighbours: list[Union[str, JID]],
        algorithms: list[str],
        camera_data: CameraData = None,
    ) -> None:
        self.jid = JID.fromstr(jid) if isinstance(jid, str) else jid
        self.password = password
        self.enable_agent_collision = enable_agent_collision
        self.prefab_name = prefab_name
        self.starter_position = starter_position
        self.behaviour_path = (
            Path(behaviour_path) if isinstance(behaviour_path, str) else behaviour_path
        )
        self.server_jid = (
            JID.fromstr(server_jid) if isinstance(server_jid, str) else server_jid
        )
        self.manager_jid = (
            JID.fromstr(manager_jid) if isinstance(manager_jid, str) else manager_jid
        )
        self.neighbours = [
            JID.fromstr(n) if isinstance(n, str) else n for n in neighbours
        ]
        self.algorithms = algorithms
        self.camera_data = camera_data
