import json


class Rgba:
    """
    RGBA (Red, Green, Blue, Alpha) stores the color intensity and the opacity between [0, 1].
    """

    def __init__(self, r: float = 0, g: float = 0, b: float = 0, a: float = 0) -> None:
        self.r: float = r
        self.g: float = g
        self.b: float = b
        self.a: float = a

    def __str__(self) -> str:
        data = {
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "a": self.a,
        }
        return json.dumps(data)
