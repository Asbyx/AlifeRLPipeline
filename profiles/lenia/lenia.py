from profiles.lenia.generator.generator import Lenia_Generator
from profiles.lenia.rewarder.rewarder import Lenia_Rewarder
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
        Load the generator, rewarder and simulation
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print("Initializing rewarder...")
        rewarder = Lenia_Rewarder(device=device)
        if os.path.exists(out_paths["rewarder"]+"/rewarder.pth"):
            print("Loading existing rewarder model...")
            rewarder.load(out_paths["rewarder"])

        print("Initializing generator...")
        generator = Lenia_Generator(rewarder)
        
        print("Initializing simulation...")
        simulation = Lenia_Simulation(generator, (400, 400), 0.1, 300, device=device)
        rewarder.set_simulation(simulation)
        
        return generator, rewarder, simulation