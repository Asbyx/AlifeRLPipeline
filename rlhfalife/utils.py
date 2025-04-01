import os
import itertools    
from typing import TYPE_CHECKING, List, Any, Callable, Tuple
if TYPE_CHECKING:
    from rlhfalife.utils import Simulator, Rewarder
    from rlhfalife.data_managers import DatasetManager, PairsManager, TrainingDataset

class Generator:
    """
    Abstract Generator class for generating parameters for an alife simulation
    Parameters are anything that can be used to configure the simulation.

    For example, for a cellular automaton, the parameters could be the initial state of the grid, or the neighborhood function.
    """

    #-------- To implement --------#
    def generate(self, nb_params: int) -> List[Any]:
        """
        Generate some parameters for the simulation.
        
        Args:
            nb_params: Number of different parameters to generate

        Returns:
            A list of parameters, of length nb_params. The parameters themselfs can be anything that can be converted to a string (override the __str__ method if needed).
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def train(self, simulator: "Simulator", rewarder: "Rewarder") -> None:
        """
        Train the generator using the rewarder

        Args:
            simulator: Simulator for which the generator is trained
            rewarder: Rewarder to train with
        """
        raise NotImplementedError("Must be implemented in inheriting class.")
    
    def save(self) -> None:
        """
        Save the generator to the path
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def load(self) -> "Generator":
        """
        Load the generator from the path

        Returns:
            The loaded generator
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    #-------- Facultative, only used for genetic search --------#
    def mutation(self, params: List[Any]) -> List[Any]:
        """
        Mutate a list of parameters.

        Args:
            params: List of parameters to mutate

        Returns:
            The mutated parameters
        """
        raise NotImplementedError("Must be implemented in inheriting class.")
    
    def crossover(self, pairs: List[Tuple[Any, Any]]) -> List[Any]:
        """
        Crossover two lists of parameters.

        Args:
            pairs: List of pairs of parameters to crossover

        Returns:
            The crossover parameters
        """
        raise NotImplementedError("Must be implemented in inheriting class.")
    
    def distance(self, pairs: List[Tuple[Any, Any]]) -> List[float]:
        """
        Distance between two lists of parameters.

        Args:
            pairs: List of pairs of parameters to compute the distance between

        Returns:
            The distances between the pairs
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    #-------- Built in --------#
    def hash_params(self, params: List[Any]) -> List[int]:
        """
        Hash a list of parameters.
        
        Note: the parameters are expected to be hashable. If not, this function is expected to be overriden in the inheriting class.

        Args:
            params: List of parameters to hash

        Returns:
            A list of hashes of the parameters
        """
        try:
            return [hash(param) for param in params]
        except:
            raise Exception("Parameters have an unhashable type. Please override the hash_params method in the Generator class.")



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
        raise NotImplementedError("Must be implemented in inheriting class.")

    def train(self, dataset: "TrainingDataset") -> None:
        """
        Train the rewarder on the pairs in pairs_manager

        Args:
            dataset: TrainingDataset instance containing the dataset.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save(self) -> None:
        """
        Save the rewarder.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def load(self) -> "Rewarder":
        """
        Load the rewarder.

        Returns:
            The loaded rewarder
        """
        raise NotImplementedError("Must be implemented in inheriting class.")
    


class Simulator:
    """
    Abstract Simulator class for the alife model
    
    It is expected to be able to run a simulation with parameters generated by a Generator.
    Parameters are anything that can be used to configure the simulation.

    For example, for a cellular automaton, the parameters could be the initial state of the grid, or the neighborhood function.
    """

    def __init__(self, generator: Generator):
        """
        Initialize the simulator with a generator.

        Args:
            generator: Generator to use for generating parameters
        """
        self.generator = generator

    #-------- To implement --------#
    def run(self, params: List[Any]) -> List[Any]:
        """
        Run the simulation with the given parameters.
        The outputs must be viewable by the Rewarder.

        Args:
            params: Parameters to run the simulation with

        Returns:
            The outputs of the simulation
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save_output(self, output: Any, path: str) -> str:
        """
        Save the output to the path.

        Args:
            output: Output to save
            path: Path to the file to save the output to. The path is the path to the file to save the output to, only the extension is expected to be specified by the user.

        Returns:
            The path to the saved output

        Example usage:
            save_output(output, "out/profile/outputs/output_number1")
            It is expected that the output will be saved in "out/profile/outputs/output_number1<extension chosen by the user>"

        Note: we do not ask for a load_output method, as the the loading of the outputs cannot be automated, since those outputs can be very big data. Therefore we let the user implement the loading in the Rewarder, from the paths given by the TrainingDataset.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save_video_from_output(self, output: Any, path: str) -> None:
        """
        Convert the output to a video and save it at path.
        It is important that the video is a .mp4 file !


        Args:
            output: Output to save
            path: Path to the file to save the video to

        Example usage:
            save_video_from_output(output, "out/profile/videos/video_number1.mp4")
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save_param(self, param: Any, path: str) -> str:
        """
        Save the param to the path.
        
        Args:
            param: Parameters of a simulation to save
            path: Path to the file to save the parameter to. The path is the path to the file to save the parameter to, only the extension is expected to be specified by the user.
        
        Example usage:
            save_param(param, "out/profile/params/param_number1")
            It is expected that the param will be saved in "out/profile/params/param_number1<extension chosen by the user>"
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def load_param(self, path: str) -> Any:
        """
        Load the param from the path.

        Args:
            path: Path to the file to load the parameter from. The path is the path to the file to load the parameter from. No need to specify the extension.

        Returns:
            The loaded parameter
        """    
        raise NotImplementedError("Must be implemented in inheriting class.")

    #-------- Built in --------#
    def generate_pairs(self, nb_params: int, dataset_manager: "DatasetManager", pairs_manager: "PairsManager", 
                      verbose: bool = False, progress_callback: Callable[[str], bool] = None) -> bool:
        """
        Generate pairs of simulations to be ranked.
        Add the new simulations to the pairs.csv for all possible pairs, including existing ones.
        
        Args:
            nb_params: Number of parameters to generate
            dataset_manager: DatasetManager instance for storing simulation data
            pairs_manager: PairsManager instance for storing pairs
            verbose: Whether to print verbose output
            progress_callback: Optional callback function to report progress
                              Should accept a message string and return True to continue, False to cancel
        """
        # Helper function to report progress
        def report_progress(message):
            if verbose: 
                print(message)
            if progress_callback:
                progress_callback(message)
        
        # Generate parameters
        report_progress("Generating parameters...")
        params = self.generator.generate(nb_params)

        # check if two params are the same
        if any(str(params[i]) == str(params[j]) for i in range(len(params)) for j in range(i+1, len(params))):
            print("\n" + "="*50)
            print("!!! WARNING !!!: Generator generated at least two identical parameters.")
            # filter out the identical parameters
            params = [params[i] for i in range(len(params)) if not any(str(params[i]) == str(params[j]) for j in range(i+1, len(params)))]
            print(f"Unique parameters: {len(params)}, over {nb_params} generated.")
            print("="*50 + "\n")

        hashs = [str(h) for h in self.generator.hash_params(params)]

        # Run simulations
        report_progress("Running simulations...")
        outputs = self.run(params)

        # Save parameters
        report_progress("Saving parameters...")
        param_paths = self.save_params(hashs, params, dataset_manager.out_paths['params'])
        
        # Save outputs
        report_progress("Saving outputs...")
        output_paths = self.save_outputs(hashs, outputs, dataset_manager.out_paths['outputs'])

        # Save videos
        report_progress("Generating and saving videos...")
        video_paths = self.save_videos(hashs, outputs, dataset_manager.out_paths['videos'])

        # Add entries to dataset manager
        report_progress("Adding entries to dataset manager...")
        dataset_manager.add_entries_from_simulation(hashs, params, outputs, param_paths, output_paths, video_paths)

        # Get all existing hashes from the dataset
        report_progress("Loading existing hashes...")
        existing_hashs = dataset_manager.get_all_hashes()

        # Generate all possible pairs of new simulations
        report_progress("Generating new pairs...")
        new_pairs = list(itertools.combinations(hashs, 2))

        # Generate pairs with existing simulations
        report_progress("Combining with existing simulations...")
        for new_hash in hashs:
            for existing_hash in existing_hashs:
                if new_hash != existing_hash and (new_hash, existing_hash) not in new_pairs and (existing_hash, new_hash) not in new_pairs:
                    new_pairs.append((new_hash, existing_hash))

        # Add new pairs to the pairs manager
        report_progress(f"Adding {len(new_pairs)} new pairs...")
        pairs_manager.add_pairs(new_pairs)
        
        return True

    def save_videos(self, hashs: List[str], outputs: List[Any], vids_path: str) -> List[str]:
        """
        Save videos from the outputs.

        Args:
            hashs: Hashes of the para   meters
            outputs: Outputs to save
            vids_path: Path to the videos folder

        Returns:
            The paths to the saved videos
        """
        res = []
        for i, output in enumerate(outputs):
            res.append(os.path.join(vids_path, f"{hashs[i]}.mp4"))
            self.save_video_from_output(output, res[-1])
        return res

    def save_outputs(self, hashs: List[str], outputs: List[Any], outputs_path: str) -> List[str]:
        """
        Save the outputs to the path.

        Args:
            hashs: Hashes of the parameters
            outputs: Outputs to save
            outputs_path: Path to the outputs folder

        Returns:
            The paths to the saved outputs
        """
        res = []
        for i, output in enumerate(outputs):
            path = os.path.join(outputs_path, f"{hashs[i]}")
            res.append(self.save_output(output, path))
        return res

    def save_params(self, hashs: List[str], params: List[Any], path: str) -> List[str]:
        """
        Save the parameters to the path.

        Args:
            hashs: Hashes of the parameters
            params: Parameters to save
            path: Path to the parameters folder

        Returns:
            The paths to the saved parameters
        """
        res = []
        for i, param in enumerate(params):
            param_path = os.path.join(path, f"{hashs[i]}")
            res.append(self.save_param(param, param_path))
        return res

class Loader:
    """
    Abstract Loader class to load a Generator, a Rewarder and a Simulator
    It is expected to be also named Loader.
    """
    def load(self, out_paths: dict, config: dict) -> tuple[Generator, Rewarder, Simulator]:
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
        raise NotImplementedError("Must be implemented in inheriting class.")


    
        
        
        
        
