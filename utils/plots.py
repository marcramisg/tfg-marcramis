import pandas as pd
import plotly.express as px


class LogData:
    """Base class for log data."""

    def __init__(self, filepath, algorithm_rounds=None):
        self.filepath = filepath
        self.algorithm_rounds = algorithm_rounds
        self.data = None

    def load_data(self):
        """Load data from CSV file into a pandas DataFrame."""
        self.data = pd.read_csv(self.filepath)
        self.convert_timestamps()
        if self.algorithm_rounds is not None:
            self.data = self.data[self.data["algorithm_round"].isin(self.algorithm_rounds)]

    def convert_timestamps(self):
        """Convert timestamp columns to datetime objects."""
        pass  # To be implemented in subclasses if needed


class AlgorithmData(LogData):
    """Class for processing algorithm data."""

    def convert_timestamps(self):
        self.data["log_timestamp"] = pd.to_datetime(self.data["log_timestamp"])
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])

    def process_data(self):
        """Compute average seconds to complete per agent and per round."""
        self.avg_time_per_agent = self.data.groupby("agent")["seconds_to_complete"].mean().reset_index()
        self.avg_time_per_round = self.data.groupby("algorithm_round")["seconds_to_complete"].mean().reset_index()

    def plot_seconds_to_complete_per_agent(self):
        """Plot average seconds to complete per agent."""
        fig = px.bar(
            self.avg_time_per_agent,
            x="agent",
            y="seconds_to_complete",
            color="agent",
            title="Average Algorithm Seconds to Complete per Agent",
        )
        fig.show()

    def plot_seconds_to_complete_per_round(self):
        """Plot average seconds to complete per algorithm round."""
        fig = px.line(
            self.avg_time_per_round,
            x="algorithm_round",
            y="seconds_to_complete",
            title="Average Algorithm Seconds to Complete per Round",
        )
        fig.show()


class MessageData(LogData):
    """Class for processing message data."""

    def convert_timestamps(self):
        self.data["log_timestamp"] = pd.to_datetime(self.data["log_timestamp"])
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])

    def process_data(self):
        """Compute message counts and size statistics."""
        self.data["is_sent"] = self.data["type"].str.startswith("SEND")
        self.sent_messages_per_round = (
            self.data[self.data["is_sent"]].groupby("algorithm_round").size().reset_index(name="sent_message_count")
        )
        self.message_counts_per_sender = self.data.groupby("sender").size().reset_index(name="counts")
        self.message_counts_per_type = self.data.groupby("type").size().reset_index(name="counts")
        self.message_size_stats = self.data["size"].describe()

    def plot_sent_messages_per_algorithm_round(self):
        """Plot number of sent messages per algorithm round."""
        fig = px.bar(
            self.sent_messages_per_round,
            x="algorithm_round",
            y="sent_message_count",
            title="Number of Sent Messages per Algorithm Round",
            labels={
                "algorithm_round": "Algorithm Round",
                "sent_message_count": "Number of Sent Messages",
            },
        )
        fig.show()

    def compute_average_size_for_type_substring(self, substring):
        """Compute average size of messages containing a specific substring in type."""
        filtered_messages = self.data[self.data["type"].str.contains(substring)]
        avg_size = filtered_messages["size"].mean()
        return avg_size

    def average_size_of_layer_messages(self):
        """Print average size of messages with 'LAYER' in type."""
        avg_size = self.compute_average_size_for_type_substring("LAYER")
        print(f"Average size of messages with 'LAYER' in type: {avg_size}")

    def average_size_of_similarity_messages(self):
        """Print average size of messages with 'SIMILARITY' in type."""
        avg_size = self.compute_average_size_for_type_substring("SIMILARITY")
        print(f"Average size of messages with 'SIMILARITY' in type: {avg_size}")


class NNInferenceData(LogData):
    """Class for processing neural network inference data."""

    def convert_timestamps(self):
        self.data["log_timestamp"] = pd.to_datetime(self.data["log_timestamp"])
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])

    def process_data(self):
        """Compute average accuracy and loss per agent."""
        self.data.sort_values("timestamp", inplace=True)
        self.data["time_in_seconds"] = (self.data["timestamp"] - self.data["timestamp"].min()).dt.total_seconds()
        self.avg_test_accuracy_per_agent = self.data.groupby("agent")["test_accuracy"].mean().reset_index()
        self.avg_test_loss_per_agent = self.data.groupby("agent")["test_loss"].mean().reset_index()

    def plot_test_accuracy_over_time_seconds(self):
        """Plot test accuracy over time in seconds."""
        fig = px.line(
            self.data,
            x="time_in_seconds",
            y="test_accuracy",
            color="agent",
            title="NN Test Accuracy Over Time (seconds)",
            labels={
                "time_in_seconds": "Time (seconds)",
                "test_accuracy": "Test Accuracy",
            },
        )
        fig.show()

    def plot_test_loss_over_time_seconds(self):
        """Plot test loss over time in seconds."""
        fig = px.line(
            self.data,
            x="time_in_seconds",
            y="test_loss",
            color="agent",
            title="NN Test Loss Over Time (seconds)",
            labels={"time_in_seconds": "Time (seconds)", "test_loss": "Test Loss"},
        )
        fig.show()


class NNTrainData(LogData):
    """Class for processing neural network training data."""

    def convert_timestamps(self):
        self.data["log_timestamp"] = pd.to_datetime(self.data["log_timestamp"])
        self.data["start_timestamp"] = pd.to_datetime(self.data["start_timestamp"])

    def process_data(self):
        """Compute average accuracy and loss per agent."""
        self.data.sort_values("start_timestamp", inplace=True)
        self.data["time_in_seconds"] = (
            self.data["start_timestamp"] - self.data["start_timestamp"].min()
        ).dt.total_seconds()
        self.avg_accuracy_per_agent = self.data.groupby("agent")["accuracy"].mean().reset_index()
        self.avg_loss_per_agent = self.data.groupby("agent")["loss"].mean().reset_index()

    def plot_accuracy_over_time_seconds(self):
        """Plot training accuracy over time in seconds."""
        fig = px.line(
            self.data,
            x="time_in_seconds",
            y="accuracy",
            color="agent",
            title="NN Train Accuracy Over Time (seconds)",
            labels={
                "time_in_seconds": "Time (seconds)",
                "accuracy": "Training Accuracy",
            },
        )
        fig.show()

    def plot_loss_over_time_seconds(self):
        """Plot training loss over time in seconds."""
        fig = px.line(
            self.data,
            x="time_in_seconds",
            y="loss",
            color="agent",
            title="NN Train Loss Over Time (seconds)",
            labels={"time_in_seconds": "Time (seconds)", "loss": "Training Loss"},
        )
        fig.show()
