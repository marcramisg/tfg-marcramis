from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ModelMetrics:
    """
    Dataclass to store various performance metrics of a model evaluation.
    """

    accuracy: float
    loss: float
    precision: None | float = None
    recall: None | float = None
    f1_score: None | float = None
    start_time_z: None | datetime = None
    end_time_z: None | datetime = None

    def time_elapsed(self) -> timedelta:
        if not self.end_time_z or not self.start_time_z:
            raise ValueError("datetime is None in time_elapsed ModelMetric class.")
        return self.end_time_z - self.start_time_z