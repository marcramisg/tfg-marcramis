import logging
from datetime import datetime, timezone

from aioxmpp import JID
from spade.template import Template

from .csv import CsvLogManager


class AlgorithmLogManager(CsvLogManager):

    def __init__(
        self,
        base_logger_name: str = "rf.algorithm",
        extra_logger_name: None | str = None,
        level: int = logging.DEBUG,
        datetime_format: str = "%Y-%m-%dT%H:%M:%S.%fZ",
        mode: str = "a",
        encoding: None | str = None,
        delay: bool = False,
    ) -> None:
        super().__init__(
            base_logger_name,
            extra_logger_name,
            level,
            datetime_format,
            mode,
            encoding,
            delay,
        )
        self.__chrono: datetime = datetime.now(tz=timezone.utc)

    @staticmethod
    def get_header() -> str:
        return "log_timestamp,log_name,algorithm_round,timestamp,agent,seconds_to_complete"

    @staticmethod
    def get_template() -> Template:
        return Template(metadata={"rf.observer.log": "algorithm"})

    def log(
        self,
        current_round: int,
        agent: JID,
        seconds: float,
        timestamp: None | datetime = None,
        level: None | int = None,
    ) -> None:
        lvl = self.level if level is None else level
        dt = datetime.now(tz=timezone.utc) if timestamp is None else timestamp
        dt_str = dt.strftime(self.datetime_format)
        agent = str(agent.bare()) if isinstance(agent, JID) else agent
        msg = ",".join(
            [
                str(current_round),
                dt_str,
                agent,
                str(seconds),
            ]
        )
        self.logger.log(level=lvl, msg=msg)

    def get_chrono_seconds(self) -> float:
        return (datetime.now(tz=timezone.utc) - self.__chrono).total_seconds()

    def restart_chrono(self) -> None:
        self.__chrono = datetime.now(tz=timezone.utc)