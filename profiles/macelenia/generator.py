from rlhfalife import Generator, Simulator, Rewarder
from typing import List, Any
from ..lenia.utils.leniaparams import LeniaParams

class LeniaGenerator(Generator):

    def __init__(self, gen_mode:str, k_size=25, device='cpu'):
        """
            Temp Generator for Lenia. This will be replaced by a better generator later.

            Args:
            gen_mode : dictionary with keys 'default_gen' (bool), 'k_arbi' (bool), 'g_arbi' (bool)
            k_size : size of the kernel
        """
        super().__init__()
        self.default_gen = gen_mode['default_gen']
        self.k_arbi = gen_mode['k_arbi']
        self.g_arbi = gen_mode['g_arbi']

        self.k_size = k_size
        self.device = device

        self.latest_rewarder = None

    def generate(self, nb_params: int) -> List[Any]:
        """
        Generate some parameters for the simulation.
        
        Args:
            nb_params: Number of different parameters to generate

        Returns:
            A list of parameters, of length nb_params. The parameters themselfs can be anything that can be converted to a string (override the __str__ method if needed).
        """
        daparams = LeniaParams.mixed_gen(batch_size=nb_params, k_size=self.k_size, device=self.device, default_base=self.default_gen, k_arbi=self.k_arbi, g_arbi=self.g_arbi)
        
        return [daparams[i] for i in range(daparams.batch_size)]
    
    def train(self, simulator: "Simulator", rewarder: "Rewarder") -> None:
        """
        Train the generator using the rewarder

        Args:
            simulator: Simulator for which the generator is trained
            rewarder: Rewarder to train with
        """
        print('POOOOF, its trained')
    
    def save(self) -> None:
        """
        Save the generator to the path
        """
        print('POOOOOOF, its saved')

    def load(self) -> "Generator":
        """
        Load the generator from the path

        Returns:
            The loaded generator
        """
        print('POOOOOOF, its loaded')
        return self