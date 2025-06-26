from aioxmpp import JID
from spade.behaviour import CyclicBehaviour
from spade.agent import Agent

from five_client.command import FiveCommand
from five_client.data import Axis


class FovCommand(FiveCommand):

    def __init__(
        self,
        fiveserver_jid: JID,
        sender_state: CyclicBehaviour,
        camera_id: int,
        fov: float,
    ) -> None:
        self.camera_id: int = camera_id
        self.fov: float = fov
        super().__init__(
            name="cameraFov",
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
        )

    @property
    def data(self) -> list[str]:
        return [str(self.camera_id), str(self.fov)]


class MoveCameraCommand(FiveCommand):
    """
    Command to move a camera along a specified axis by a relative position.
    """

    def __init__(
        self,
        fiveserver_jid: JID,
        sender_state: CyclicBehaviour,
        camera_id: int,
        axis: Axis,
        relative_position: float,
    ) -> None:
        """
        Initializes a MoveCameraCommand instance.

        Args:
            fiveserver_jid (JID): The JID of the FIVE server.
            sender_state (CyclicBehaviour): The sender state to handle message sending.
            camera_id (int): The ID of the camera to be moved.
            axis (Axis): The axis along which the camera will be moved.
            relative_position (float): The relative position by which to move the camera.
        """
        self.camera_id: int = camera_id
        self.axis: Axis = axis
        self.relative_position: float = relative_position
        super().__init__(
            name="cameraMove",
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
        )

    @property
    def data(self) -> list[str]:
        return [str(self.camera_id), str(self.axis), str(self.relative_position)]


class RotateCameraCommand(FiveCommand):
    """
    Command to rotate a camera along a specified axis by a specified number of degrees.
    """

    def __init__(
        self,
        fiveserver_jid: JID,
        sender_state: CyclicBehaviour,
        camera_id: int,
        axis: Axis,
        degrees: float,
    ) -> None:
        """
        Initializes a RotateCameraCommand instance.

        Args:
            fiveserver_jid (JID): The JID of the FIVE server.
            sender_state (CyclicBehaviour): The sender state to handle message sending.
            camera_id (int): The ID of the camera to be moved.
            axis (Axis): The axis along which the camera will be moved.
            degrees (float): The number of degrees by which to rotate the camera.
        """
        self.camera_id: int = camera_id
        self.axis: Axis = axis
        self.degrees: float = degrees
        super().__init__(
            name="cameraRotate",
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
        )

    @property
    def data(self) -> list[str]:
        return [str(self.camera_id), str(self.axis), str(self.degrees)]


class CaptureImageCommand(FiveCommand):

    def __init__(
        self,
        fiveserver_jid: JID,
        sender_state: CyclicBehaviour,
        camera_id: int,
        image_mode: float,
    ) -> None:
        self.camera_id: int = camera_id
        self.image_mode: float = image_mode
        super().__init__(
            name="image",
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
        )

    @property
    def data(self) -> list[str]:
        return [str(self.camera_id), str(self.image_mode)]


class SingleImageCaptureCommand(CaptureImageCommand):

    def __init__(
        self, fiveserver_jid: JID, sender_state: CyclicBehaviour, camera_id: int
    ) -> None:
        self.camera_id: int = camera_id
        super().__init__(
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
            camera_id=self.camera_id,
            image_mode=0,
        )


class PeriodicImageCaptureCommand(CaptureImageCommand):

    def __init__(
        self,
        fiveserver_jid: JID,
        sender_state: CyclicBehaviour,
        camera_id: int,
        frequency: float,
    ) -> None:
        self.camera_id: int = camera_id
        if frequency <= 0:
            agent: Agent = sender_state.agent
            raise ValueError(
                f"In the PeriodicImageCaptureCommand class of agent {agent.jid}, frequency must be greater than zero, but instead it is {frequency:.2f}."
            )
        super().__init__(
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
            camera_id=self.camera_id,
            image_mode=frequency,
        )


class StopPeriodicImageCaptureCommand(CaptureImageCommand):

    def __init__(
        self, fiveserver_jid: JID, sender_state: CyclicBehaviour, camera_id: int
    ) -> None:
        self.camera_id: int = camera_id
        super().__init__(
            fiveserver_jid=fiveserver_jid,
            sender_state=sender_state,
            camera_id=self.camera_id,
            image_mode=-1,
        )
