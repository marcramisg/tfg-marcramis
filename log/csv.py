import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path

from spade.template import Template


class CsvLogManager(object, metaclass=ABCMeta):
    def __init__(
        self,
        base_logger_name: str,
        extra_logger_name: None | str = None,
        level: int = logging.DEBUG,
        datetime_format: str = "%Y-%m-%dT%H:%M:%S.%fZ",
        mode: str = "a",
        encoding: None | str = None,
        delay: bool = False,
    ) -> None:
        self.base_logger_name = base_logger_name
        self._extra_logger_name = extra_logger_name
        self.level = level
        self.datetime_format = datetime_format
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        self.formatter = logging.Formatter("%(asctime)s.%(msecs)03d,%(name)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        self.logger = self.__get_logger()

    def __get_logger(self) -> logging.Logger:
        logger_name = (
            self.base_logger_name
            if self._extra_logger_name is None
            else f"{self.base_logger_name}.{self._extra_logger_name}"
        )
        return logging.getLogger(logger_name)

    @property
    def extra_logger_name(self) -> str | None:
        return self._extra_logger_name

    @extra_logger_name.setter
    def extra_logger_name(self, value: str) -> None:
        self._extra_logger_name = value
        self.__get_logger()

    def setup(self, folder_name: str | Path, file_name: str | Path) -> None:
        log_path = Path(folder_name)
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
        log_path = log_path / file_name
        base_logger = logging.getLogger(self.base_logger_name)
        if len(base_logger.handlers) == 0:
            csv_handler = CsvFileHandler(
                path=log_path,
                header=self.get_header(),
                mode=self.mode,
                encoding=self.encoding,
                delay=self.delay,
            )
            csv_handler.setLevel(self.level)
            csv_handler.setFormatter(self.formatter)
            base_logger.setLevel(self.level)
            base_logger.addHandler(csv_handler)

    @staticmethod
    @abstractmethod
    def get_header() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_template() -> Template:
        raise NotImplementedError


class CsvFileHandler(logging.FileHandler):
    def __init__(
        self,
        path: Path,
        header: str,
        mode: str = "a",
        encoding: None | str = None,
        delay: bool = False,
    ):
        self.header = header
        file_exists = path.exists() and path.is_file()
        super().__init__(path, mode=mode, encoding=encoding, delay=delay)
        if not file_exists or mode == "w" or path.stat().st_size == 0:
            if not self.stream:
                self.stream = self._open()
            self.stream.write(self.header + "\n")
            self.stream.flush()