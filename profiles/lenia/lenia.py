from profiles.lenia.generator.generator import Lenia_Generator
from profiles.lenia.rewardor.rewardor import Lenia_Rewardor
from profiles.lenia.simulation.simulation import Lenia_Simulation
import src.utils
import torch

class Loader(src.utils.Loader):
    """
        Loader for the Lenia-like automaton.
    """
    def load(self):
        """
        Load the generator, rewardor and simulation
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Create generator and simulation
        print("Initializing rewardor...")
        rewardor = Lenia_Rewardor(device=device)
        print("Initializing generator...")
        generator = Lenia_Generator(rewardor)
        print("Initializing simulation...")
        simulation = Lenia_Simulation(generator, (3, 400, 400), 0.1, 300)
        rewardor.set_simulation(simulation)
        return generator, rewardor, simulation