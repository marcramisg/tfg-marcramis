import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .algorithm import AlgorithmLogManager
from .general import GeneralLogManager
from .message import MessageLogManager
from .nn import NnConvergenceLogManager, NnInferenceLogManager, NnTrainLogManager


def setup_loggers(
    log_folder_path: str | Path = "logs",
    datetime_mark: bool = True,
    general_level: int = logging.DEBUG,
    csv_level: int = logging.DEBUG,
) -> None:
    log_folder = Path(log_folder_path)
    if datetime_mark:
        log_folder = log_folder / datetime.now(tz=timezone.utc).strftime("%Y_%m_%d_T_%H_%M_%S_%f_Z") / "raw"
    else:
        log_folder = log_folder / str(uuid.uuid4()).replace("-", "_")

    GeneralLogManager(level=general_level).setup(folder_name=log_folder, file_name="general.log")
    AlgorithmLogManager(level=csv_level).setup(folder_name=log_folder, file_name="algorithm.csv")
    NnInferenceLogManager(level=csv_level).setup(folder_name=log_folder, file_name="nn_inference.csv")
    NnTrainLogManager(level=csv_level).setup(folder_name=log_folder, file_name="nn_train.csv")
    NnConvergenceLogManager(level=csv_level).setup(folder_name=log_folder, file_name="nn_convergence.csv")
    MessageLogManager(level=csv_level).setup(folder_name=log_folder, file_name="message.csv")