"""
Factory class to create a list of FL agents based on an Experiment and graph manager.
"""

from typing import List
from uuid import UUID

from aioxmpp import JID

from ..datatypes.consensus_manager import ConsensusManager
from ..datatypes.data import DatasetSettings, IidDatasetSettings, NonIidDirichletDatasetSettings
from ..datatypes.experiment import ExperimentRawData
from ..datatypes.graph import GraphManager
from ..datatypes.models import ModelManager
from ..nn.model_factory import ModelManagerFactory
from ..similarity.function import EuclideanDistanceFunction
from ..similarity.similarity_manager import SimilarityManager
from .agent import PremioFlAgent
from .premiofl.acol import AcolAgent
from .premiofl.macofl import MacoflAgent
from .premiofl.pmacofl_max import PmacoflMaxAgent
from .premiofl.pmacofl_min import PmacoflMinAgent


def create_dataset_settings(distribution: str, num_clients: int, client_index: int) -> DatasetSettings:
    """
    Create either IID or Non-IID dataset settings, depending on the distribution field.
    For example, if distribution is 'iid', use IidDatasetSettings; if 'non_iid diritchlet ...',
    use NonIidDirichletDatasetSettings. You can parse 'alpha' from distribution if needed.
    """
    dist_lower = distribution.lower()
    if dist_lower.startswith("iid"):
        return IidDatasetSettings(
            seed=13,
            train_samples_percent=0.1,
            test_samples_percent=1.0,
        )
    if dist_lower.startswith("non_iid diritchlet"):
        alpha = float(dist_lower.split("non_iid diritchlet")[1].strip())
        return NonIidDirichletDatasetSettings(
            seed=13,
            num_clients=num_clients,
            client_index=client_index,
            dirichlet_alpha=alpha,
        )
    raise NotImplementedError("Distribution of dataset does not exist.")


def create_experiment_agent(
    algorithm: str,
    jid: str,
    password: str,
    max_message_size: int,
    consensus_manager: ConsensusManager,
    model_manager: ModelManager,
    similarity_manager: SimilarityManager,
    observers: List[JID],
    neighbours: List[JID],
    coordinator: JID,
    max_rounds: int = 70,
    verify_security: bool = False,
) -> PremioFlAgent:
    if algorithm.lower() == "acol":
        return AcolAgent(
            jid=jid,
            password=password,
            max_message_size=max_message_size,
            consensus_manager=consensus_manager,
            model_manager=model_manager,
            similarity_manager=similarity_manager,
            observers=observers,
            neighbours=neighbours,
            coordinator=coordinator,
            max_rounds=max_rounds,
            verify_security=verify_security,
        )
    if algorithm.lower() == "macofl":
        return MacoflAgent(
            jid=jid,
            password=password,
            max_message_size=max_message_size,
            consensus_manager=consensus_manager,
            model_manager=model_manager,
            similarity_manager=similarity_manager,
            observers=observers,
            neighbours=neighbours,
            coordinator=coordinator,
            max_rounds=max_rounds,
            verify_security=verify_security,
        )
    if algorithm.lower() == "pmacofl_min":
        return PmacoflMinAgent(
            jid=jid,
            password=password,
            max_message_size=max_message_size,
            consensus_manager=consensus_manager,
            model_manager=model_manager,
            similarity_manager=similarity_manager,
            observers=observers,
            neighbours=neighbours,
            coordinator=coordinator,
            max_rounds=max_rounds,
            verify_security=verify_security,
        )
    if algorithm.lower() == "pmacofl_max":
        return PmacoflMaxAgent(
            jid=jid,
            password=password,
            max_message_size=max_message_size,
            consensus_manager=consensus_manager,
            model_manager=model_manager,
            similarity_manager=similarity_manager,
            observers=observers,
            neighbours=neighbours,
            coordinator=coordinator,
            max_rounds=max_rounds,
            verify_security=verify_security,
        )
    raise NotImplementedError(f"Algorithm {algorithm.lower()} not recognized.")


class AgentFactory:
    """
    A factory to build a list of FL agents based on:
      - The experiment (algorithm, distribution, dataset, etc.).
      - The GraphManager (for neighbor relationships).
      - Global settings like coordinator JID, observer JIDs, etc.
    """

    def __init__(
        self,
        experiment: ExperimentRawData,
        graph_manager: GraphManager,
        coordinator_jid: JID,
        observer_jids: List[JID],
        uuid: UUID | None = None,
        max_message_size: int = 250_000,
        verify_security: bool = False,
    ) -> None:
        self.experiment = experiment
        self.graph_manager = graph_manager
        self.coordinator_jid = coordinator_jid
        self.observer_jids = observer_jids
        self.uuid = uuid
        self.max_message_size = max_message_size
        self.verify_security = verify_security

    def create_agents(self, dummy) -> List[PremioFlAgent]:
        """
        Create and return a list of FL agents based on the Experiment data and graph structure.
        """
        algorithm_rounds = self.experiment.algorithm_rounds or 120
        consensus_iterations = self.experiment.consensus_iterations or 10

        agents: List[PremioFlAgent] = []
        agent_localparts = self.graph_manager.list_agents_with_neighbours(uuid=self.uuid)
        max_order = max(len(ns) for ns in agent_localparts.values())
        for index, localpart in enumerate(agent_localparts.keys()):
            agent_jid = JID.fromstr(f"{localpart}@{self.experiment.xmpp_domain}")
            neighbor_localparts = agent_localparts[localpart]
            neighbours = [JID.fromstr(f"{n}@{self.experiment.xmpp_domain}") for n in neighbor_localparts]

            # Create dataset settings
            dataset_settings = create_dataset_settings(
                self.experiment.distribution,
                num_clients=len(agent_localparts),
                client_index=index,
            )

            # Create model manager
            model_manager = ModelManagerFactory.get_manager(
                dataset=self.experiment.dataset,
                settings=dataset_settings,
                ann=self.experiment.ann,
                training_epochs=self.experiment.training_epochs,
                dummy=dummy,
            )

            # Create consensus manager
            consensus = ConsensusManager(
                model_manager=model_manager,
                max_order=max_order,
                max_seconds_to_accept_consensus=24 * 60 * 60,
                consensus_iterations=consensus_iterations,
            )

            # Create similarity manager
            similarity_manager = SimilarityManager(
                model_manager=model_manager,
                function=EuclideanDistanceFunction(),
                wait_for_responses_timeout=5 * 60,
            )

            # Instantiate the FL agent
            agent = create_experiment_agent(
                algorithm=self.experiment.algorithm,
                jid=str(agent_jid.bare()),
                password="123",
                max_message_size=self.max_message_size,
                consensus_manager=consensus,
                model_manager=model_manager,
                similarity_manager=similarity_manager,
                observers=self.observer_jids,
                neighbours=neighbours,
                coordinator=self.coordinator_jid,
                max_rounds=algorithm_rounds,
                verify_security=self.verify_security,
            )
            agents.append(agent)

        return agents
