import json

from abc import ABCMeta


class PositionData(metaclass=ABCMeta):
    """
    Abstract class to represent the position of the agent.
    """

    def __init__(self, position: dict[str, float]) -> None:
        self.__position: dict[str, float] = position

    def __str__(self) -> str:
        return json.dumps(self.__position)


class DigitalPositionData(PositionData):
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.__position: dict[str, float] = {"x": x, "y": y, "z": z}
        super().__init__(self.__position)


class GeographicPositionData(PositionData):
    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.__position: dict[str, float] = {
            "latitude": latitude,
            "longitude": longitude,
        }
        super().__init__(self.__position)
