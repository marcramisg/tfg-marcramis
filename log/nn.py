import logging
from datetime import datetime, timezone

from aioxmpp import JID
from spade.template import Template

from ..datatypes.metrics import ModelMetrics
from .csv import CsvLogManager


class NnInferenceLogManager(CsvLogManager):

    def __init__(
        self,
        base_logger_name="rf.nn.inference",
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
        return "log_timestamp,log_name,algorithm_round,timestamp,agent,seconds_to_complete,epochs,mean_training_accuracy,mean_training_loss,validation_accuracy,validation_loss,test_accuracy,test_loss"

    @staticmethod
    def get_template() -> Template:
        return Template(metadata={"rf.observer.log": "nn.inference"})

    def log(
        self,
        current_round: int,
        agent: str | JID,
        seconds: float,
        epochs: int,
        mean_training_accuracy: float,
        mean_training_loss: float,
        validation_accuracy: float,
        validation_loss: float,
        test_accuracy: float,
        test_loss: float,
        timestamp: None | datetime = None,
        level: None | int = logging.DEBUG,
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
                str(epochs),
                str(mean_training_accuracy),
                str(mean_training_loss),
                str(validation_accuracy),
                str(validation_loss),
                str(test_accuracy),
                str(test_loss),
            ]
        )
        self.logger.log(level=lvl, msg=msg)


class NnTrainLogManager(CsvLogManager):

    def __init__(
        self,
        base_logger_name="rf.nn.train",
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
        return "log_timestamp,log_name,algorithm_round,start_timestamp,agent,seconds_to_complete,epoch,accuracy,loss"

    @staticmethod
    def get_template() -> Template:
        return Template(metadata={"rf.observer.log": "nn.train"})

    def log(
        self,
        current_round: int,
        agent: str | JID,
        seconds: float,
        epoch: int,
        accuracy: float,
        loss: float,
        start_timestamp: None | datetime = None,
        level: None | int = logging.DEBUG,
    ) -> None:
        lvl = self.level if level is None else level
        dt = datetime.now(tz=timezone.utc) if start_timestamp is None else start_timestamp
        dt_str = dt.strftime(self.datetime_format)
        agent = str(agent.bare()) if isinstance(agent, JID) else agent
        msg = ",".join(
            [
                str(current_round),
                dt_str,
                agent,
                str(seconds),
                str(epoch),
                str(accuracy),
                str(loss),
            ]
        )
        self.logger.log(level=lvl, msg=msg)

    def log_train_epoch(self, epoch: int, train: ModelMetrics, agent_jid: JID, current_round: int) -> None:
        if train.start_time_z is not None and train.end_time_z is not None:
            time = train.end_time_z - train.start_time_z
            self.log(
                current_round=current_round,
                agent=agent_jid,
                seconds=time.total_seconds(),
                epoch=epoch,
                accuracy=train.accuracy,
                loss=train.loss,
                start_timestamp=train.start_time_z,
            )


class NnConvergenceLogManager(CsvLogManager):

    def __init__(
        self,
        base_logger_name="rf.nn.convergence",
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
        return "log_timestamp,log_name,algorithm_round,agent,epoch,layer,weight"

    @staticmethod
    def get_template() -> Template:
        return Template(metadata={"rf.observer.log": "nn.convergence"})

    def log(
        self,
        current_round: int,
        agent: str | JID,
        epoch: int,
        layer: str,
        weight: float,
        level: None | int = logging.DEBUG,
    ) -> None:
        lvl = self.level if level is None else level
        agent = str(agent.bare()) if isinstance(agent, JID) else agent
        msg = ",".join(
            [
                str(current_round),
                agent,
                str(epoch),
                layer,
                str(weight),
            ]
        )
        self.logger.log(level=lvl, msg=msg)

    def log_weights(self, epoch: int, layer: str, weight: float, agent_jid: JID, current_round: int) -> None:
        self.log(current_round=current_round, agent=agent_jid, epoch=epoch, layer=layer, weight=weight)