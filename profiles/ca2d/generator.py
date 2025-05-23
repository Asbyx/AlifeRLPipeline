from rlhfalife import Generator, Simulator, Rewarder
from typing import List, Any
import random


class CA2DGenerator(Generator):
    """
    Abstract Generator class for generating parameters for an alife simulation
    Parameters are anything that can be used to configure the simulation.

    For example, for a cellular automaton, the parameters could be the initial state of the grid, or the neighborhood function.
    """

    def __init__(self, seed=None, device='cpu'):
        """
        Initialize the generator for CA2D parameters

        Args:
            seed: Seed for the random number generator, for reproducibility.
            device: Device to run the simulation on. Defaults to "cpu". If "cuda" is available, it will be used.
        """
        super().__init__()
        self.seed = seed
        if(seed is not None):
            random.seed(seed)
        self.device = device
        self.latest_rewarder = None
        self.latest_simulator = None
    #-------- To implement --------#
    def generate(self, nb_params: int) -> List[Any]:
        """
        Generate some parameters for the simulation.
        
        Args:
            nb_params: Number of different parameters to generate

        Returns:
            A list of parameters, of length nb_params. The parameters themselfs can be anything that can be converted to a string (override the __str__ method if needed).
        """
        sampled_params = []

        for _ in range(nb_params):
            s_num = random.randint(0, 2**9-1)
            b_num = random.randint(0, 2**9-1)
            sampled_params.append((s_num, b_num))
    
        return sampled_params

    def reward_generate(self, nb_params: int, score_threshold = 10.,  rewarder_path=None) -> List[Any]:
        """
            Generate parameters, and keep only the ones that are above the score threshold.
        """
    def train(self, simulator: "Simulator", rewarder: "Rewarder") -> None:
        """
        'Trains' the generator : stores the latest rewarder for later use

        Args:
            simulator: Simulator for which the generator is trained (useless)
            rewarder: Trained rewarder
        """
        self.latest_rewarder = rewarder
    
    def save(self) -> None:
        """
        Save the generator to the path
        """
        print('Saving not implemented yet, but it would essentially just be saving the rewarder')

    def load(self) -> "Generator":
        """
        Load the generator from the path

        Returns:
            The loaded generator
        """
        print('Loading not implemented yet, but it would essentially just be loading the rewarder')