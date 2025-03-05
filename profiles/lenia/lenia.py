from profiles.lenia.generator.generator import Lenia_Generator
from profiles.lenia.rewardor.rewardor import Lenia_Rewardor
from profiles.lenia.simulation.simulation import Lenia_Simulation
import src.utils
import torch
import os
class Loader(src.utils.Loader):
    """
        Loader for Lenia.
    """
    def load(self, out_paths):
        """
        Load the generator, rewardor and simulation
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print("Initializing rewardor...")
        rewardor = Lenia_Rewardor(device=device)
        if os.path.exists(out_paths["rewardor"]):
            print("Loading existing rewardor model...")
            rewardor.load(out_paths["rewardor"])

        print("Initializing generator...")
        generator = Lenia_Generator(rewardor)
        
        print("Initializing simulation...")
        simulation = Lenia_Simulation(generator, (400, 400), 0.1, 300, device=device)
        rewardor.set_simulation(simulation)
        
        return generator, rewardor, simulation