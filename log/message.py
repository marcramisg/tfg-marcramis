import logging
from datetime import datetime, timezone

from aioxmpp import JID
from spade.template import Template

from .csv import CsvLogManager


class MessageLogManager(CsvLogManager):

    def __init__(
        self,
        base_logger_name="rf.message",
        extra_logger_name=None,
        level=logging.DEBUG,
        datetime_format="%Y-%m-%dT%H:%M:%S.%fZ",
        mode="a",
        encoding=None,
        delay=False,
    ):
        super().__init__(
            base_logger_name,
            extra_logger_name,
            level,
            datetime_format,
            mode,
            encoding,
            delay,
        )

    @staticmethod
    def get_header() -> str:
        return "log_timestamp,log_name,algorithm_round,timestamp,sender,to,type,size,thread"

    @staticmethod
    def get_template() -> Template:
        return Template(metadata={"rf.observer.log": "message"})

    def log(
        self,
        current_round: int,
        sender: str | JID,
        to: str | JID,
        msg_type: str,
        size: int,
        thread: None | str = None,
        timestamp: None | datetime = None,
        level: None | int = None,
    ) -> None:
        lvl = self.level if level is None else level
        dt = datetime.now(tz=timezone.utc) if timestamp is None else timestamp
        dt_str = dt.strftime(self.datetime_format)
        sender = str(sender.bare()) if isinstance(sender, JID) else sender
        to = str(to.bare()) if isinstance(to, JID) else to
        thread = "" if thread is None else thread
        msg = ",".join([str(current_round), dt_str, sender, to, msg_type, str(size), thread])
        self.logger.log(level=lvl, msg=msg)