from .generator.generator import Lenia_Generator
from .rewarder.rewarder import Lenia_Rewarder
from .simulator.simulator import Lenia_Simulator
import rlhfalife.utils
import torch
import os

class Loader(rlhfalife.utils.Loader):
    """
        Loader for Lenia.
    """
    def load(self, out_paths):
        """
        Load the generator, rewarder and simulator
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print("Initializing rewarder...")
        rewarder = Lenia_Rewarder(device=device)
        if os.path.exists(out_paths["rewarder"]+"/rewarder.pth"):
            print("Loading existing rewarder model...")
            rewarder.load(out_paths["rewarder"])

        print("Initializing generator...")
        generator = Lenia_Generator(rewarder)
        
        print("Initializing simulator...")
        simulator = Lenia_Simulator(generator, (400, 400), 0.1, 300, device=device)
        rewarder.set_simulator(simulator)
        
        return generator, rewarder, simulator