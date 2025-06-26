"""Experiment model for the RoyalFlush application."""

import uuid
from typing import Any, Dict, Optional


class Experiment:
    # TODO clean the ExperimentRawData to store the processed values here
    def __init__(self, uuid4: Optional[str]):
        self.uuid4 = str(uuid.uuid4()) if uuid4 == "generate_new_uuid4" else uuid4


class ExperimentRawData:
    """
    Represents an experiment configuration.

    The algorithm is made case-insensitive by storing it as lowercase internally.

    Attributes:
        algorithm (str): The name of the algorithm, stored in lowercase for uniformity.
        algorithm_rounds (int): Number of algorithm rounds.
        consensus_iterations (int): Number of consensus iterations.
        training_epochs (int): Number of training epochs.
        xmpp_domain (str): Domain name of the XMPP server.
        graph_path (str): Path to the graph file.
        dataset (str): The name of the dataset.
        distribution (str): Distribution settings (e.g. 'non_iid diritchlet 0.1' or 'iid').
        ann (str): Neural network architecture (e.g. 'cnn5', 'mlp', etc.).
        uuid4 (str | None): Can be:
            - None, if the agents do not use a uuid4 part in their names,
            - "generate_new_uuid4", if the experiment will generate new UUID4,
            - or a specific UUID4 literal.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        """
        Initialize an Experiment instance using a dictionary of fields.

        Args:
            data (Dict[str, Any]): A dictionary loaded from a JSON file that contains
                the fields for the experiment.
        """
        self.uuid4: Optional[str] = data.get("uuid4", None)
        self.algorithm: str = data.get("algorithm", "").lower()
        self.algorithm_rounds: int = data.get("algorithm_rounds", 0)
        self.consensus_iterations: int = data.get("consensus_iterations", 0)
        self.training_epochs: int = data.get("training_epochs", 0)
        self.xmpp_domain: str = data.get("xmpp_domain", "localhost")
        self.graph_path: str = data.get("graph_path", "")
        self.dataset: str = data.get("dataset", "")
        self.distribution: str = data.get("distribution", "")
        self.ann: str = data.get("ann", "")

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "ExperimentRawData":
        """
        Create an Experiment instance directly from a JSON dictionary.

        Args:
            json_data (Dict[str, Any]): A dictionary containing experiment data.

        Returns:
            Experiment: A new Experiment instance.
        """
        return cls(json_data)

    def __repr__(self) -> str:
        """Return a human-readable representation of the Experiment."""
        return (
            f"<Experiment uuid4={self.uuid4}, "
            f"algorithm={self.algorithm}, "
            f"algorithm_rounds={self.algorithm_rounds}, "
            f"consensus_iterations={self.consensus_iterations}, "
            f"training_epochs={self.training_epochs}, "
            f"xmpp_domain={self.xmpp_domain}, "
            f"graph_path={self.graph_path}, "
            f"dataset={self.dataset}, "
            f"distribution={self.distribution}, "
            f"ann={self.ann}>"
        )
