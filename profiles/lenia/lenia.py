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
    def load(self, out_paths, config):
        """
        Load the generator, rewarder and simulator
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print("Initializing rewarder...")
        rewarder = Lenia_Rewarder(device=device, save_path=out_paths["rewarder"])
        if os.path.exists(out_paths["rewarder"]+"/rewarder.pth"):
            print("Loading existing rewarder model...")
            rewarder.load(out_paths["rewarder"])

        print("Initializing generator...")
        generator = Lenia_Generator(rewarder)
        
        print("Initializing simulator...")
        size = config["simulation"]["width"], config["simulation"]["height"]
        dt = config["simulation"]["dt"]
        t_max = config["simulation"]["t_max"]
        simulator = Lenia_Simulator(generator, size, dt, t_max, device=device)
        rewarder.set_simulator(simulator)
        
        return generator, rewarder, simulator