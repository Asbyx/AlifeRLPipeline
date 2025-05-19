from .generator import LeniaGenerator
from ..lenia.rewarder import LeniaRewarder
from ..lenia.simulator import LeniaSimulator
import rlhfalife.utils
import os
from pathlib import Path
from .mace_lenia import MaCELenia

class Loader(rlhfalife.utils.Loader):
    """
        Loader for Lenia.
    """
    def load(self, out_paths, config):
        """
        Load the generator, rewarder and simulator

        Args:
            out_paths: Dictionary containing: 
                - 'outputs': path to the outputs folder,
                - 'videos': path to the videos folder,
                - 'params': path to the params folder,
                - 'rewarder': path to the rewarders folder,
                - 'generator': path to the generators folder,
                - 'saved_simulations': path to the saved simulations folder,
            config: Dictionary containing the config of the experiment
        """
        device = config['device']
        print("Initializing generator...")
        generator = LeniaGenerator(**config["generator"],device=device)

        print("Initializing simulator...")
        size = config["simulator"]["size"]
        channels = config["simulator"]["num_channels"]
        run_length = config["simulator"]["run_length"]	
        frame_portion = config["simulator"]["frame_portion"]

        lenia = MaCELenia(size,dt=0.1,num_channels=channels, device=device)
        lenia._beta  = 7.

        simulator = LeniaSimulator(generator=generator,init_circle=True, lenia_automaton=lenia,run_length=run_length,frame_portion=frame_portion,device=device)

        print("Initializing rewarder...")
        rewarder = LeniaRewarder(config=config, model_path=Path(out_paths["rewarder"])/f"{config["name"]}.pt", device=device, simulator=simulator, wandb_params= {"project": 'rlalife', "name": config["name"]})
        if os.path.exists(Path(out_paths["rewarder"]) / f"{config["name"]}.pt"):
            print("Loading existing rewarder model...")
            rewarder.load()


        

        
        return generator, rewarder, simulator