class ReputationManager:
    """Manages reputation scores and calculates trust for agents.

    This class provides methods to get and update reputation scores for agents,
    as well as to calculate trust scores based on peer feedback.

    Attributes:
        reputation_scores (dict[str, int]): A dictionary mapping agent IDs to their
            respective reputation scores.
        default_reputation (int): The default reputation score assigned to new agents.
    """

    def __init__(self, initial_reputation: int = 50) -> None:
        """Initializes the ReputationManager with an optional initial reputation.

        Args:
            initial_reputation: The default reputation score assigned to new agents.
                Defaults to 50.
        """
        self.reputation_scores: dict[str, int] = {}
        self.default_reputation: int = initial_reputation

    def get_reputation(self, agent_id: str) -> int:
        """Retrieves the reputation score for a given agent.

        Args:
            agent_id: The unique identifier of the agent.

        Returns:
            The reputation score of the agent. If the agent is not found,
            returns the default reputation score.
        """
        return self.reputation_scores.get(agent_id, self.default_reputation)

    def update_reputation(self, agent_id: str, success: bool = True, feedback: int = 0) -> None:
        """Updates the reputation score for a given agent based on interaction outcome.

        If the agent does not have an existing reputation score, it is initialized
        with the default reputation.

        Args:
            agent_id: The unique identifier of the agent.
            success: Indicates whether the interaction was successful. Defaults to True.
            feedback: Additional feedback score to adjust the reputation. Defaults to 0.
        """
        if agent_id not in self.reputation_scores:
            self.reputation_scores[agent_id] = self.default_reputation

        if success:
            self.reputation_scores[agent_id] += 5 + feedback
        else:
            self.reputation_scores[agent_id] -= 10 - feedback

        # Ensure the reputation score remains within the range [0, 100]
        self.reputation_scores[agent_id] = max(0, min(self.reputation_scores[agent_id], 100))

    def calculate_trust(self, agent_id: str, peer_feedback: int = 0) -> int:
        """Calculates the trust score for a given agent based on peer feedback.

        The trust score is a weighted combination of the agent's reputation score
        and peer feedback.

        Args:
            agent_id: The unique identifier of the agent.
            peer_feedback: Feedback score from peers to adjust the trust calculation.
                Defaults to 0.

        Returns:
            The calculated trust score, normalized to be within the range [0, 100].
        """
        reputation_score = self.get_reputation(agent_id)
        trust_score = 0.7 * reputation_score + 0.3 * peer_feedback
        return max(0, min(int(trust_score), 100))
