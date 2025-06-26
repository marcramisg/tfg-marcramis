from pathlib import Path


class CameraData:
    def __init__(self, maximum_images: int, image_folder: str) -> None:
        self.image_folder: Path = Path(image_folder).resolve()
        self.maximum_images: int = maximum_images
