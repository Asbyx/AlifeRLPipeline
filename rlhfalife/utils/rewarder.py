from typing import List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_managers import TrainingDataset

class Rewarder:
    """
    Abstract Rewarder class for estimating the reward
    It is expected to be trained on a dataset of pairs of simulations, with the winner of each pair, and be used to train a Generator.
    """

    #-------- To implement --------#
    def rank(self, data: List[Any]) -> List[float]:
        """
        Rank the data. 
        
        Args:
            data: Data to rank, array-like of outputs of a Simulator.

        Returns:
            An array-like of the same length as data, where the i-th element is the reward for the i-th sample.
        """
        raise NotImplementedError("Rewarder.rank must be implemented in inheriting class")

    def train(self, dataset: "TrainingDataset") -> None:
        """
        Train the rewarder on the pairs in pairs_manager

        Args:
            dataset: TrainingDataset instance containing the dataset.
        """
        raise NotImplementedError("Rewarder.train must be implemented in inheriting class")

    def save(self) -> None:
        """
        Save the rewarder.
        """
        raise NotImplementedError("Rewarder.save must be implemented in inheriting class")

    def load(self) -> "Rewarder":
        """
        Load the rewarder.

        Returns:
            The loaded rewarder
        """
        raise NotImplementedError("Rewarder.load must be implemented in inheriting class") 