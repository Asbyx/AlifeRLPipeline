import os
import pandas as pd
import itertools

class Loader:
    """
    Abstract Loader class to load a Generator, a Rewardor and a Simulation
    It is expected to be also named Loader.
    """
    def load(self, out_paths):
        """
        Load the generator, rewardor and simulation
        """
        raise NotImplementedError("Must be implemented in inheriting class.")    
    

class Generator:
    """Abstract Generator class for generating parameters for a alife model"""
    def generate(self, nb_params):
        """
        Generate some parameters
        The result must be a list of anything, of length nb_params.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def train(self, simulation, rewardor):
        """Train the model using the rewardor"""
        raise NotImplementedError("Must be implemented in inheriting class.")
    
    def hash_params(self, params):
        """Hash a list of parameters"""
        res = []
        for param in params:
            res.append(hash(param))
        return res

    def save(self, path):
        """Save the generator to the path"""
        raise NotImplementedError("Must be implemented in inheriting class.")

    def load(self, path):
        """Load the generator from the path"""
        raise NotImplementedError("Must be implemented in inheriting class.")



class Rewardor:
    """Abstract Rewardor class for estimating the reward"""
    def rank(self, data):
        """
        Rank the data. 
        data is in shape: (B, **data), where n is the number of samples, and the rest is for the data outputed by the simulation. 

        Returns a tensor of shape (B, 1), where the i-th element is the reward for the i-th sample.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def train(self, pairs_path, out_path):
        """
        Train the rewardor
        It is expected to be trained on the file pairs_path, which is a csv file with the following columns: param1, param2, winner.
        Beware of null values in the winner column, it is expected to be many of them.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save(self, path):
        """
        Save the rewardor to the path.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def load(self, path):
        """
        Load the rewardor from the path.

        Returns the loaded rewardor.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")
    


class Simulation:
    """Abstract Simulation class for the alife model"""

    def __init__(self, generator):
        self.generator = generator

    #-------- To implement --------#
    def run(self, params):
        """
        Run the simulation with the given parameters.
        The outputs must be viewable by the Rewardor.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save_output(self, output, output_path):
        """
        Save the output to the output_path.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save_video_from_output(self, output, vid_path):
        """
        Convert the output to a video and save it at vid_path.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save_params(self, params, params_path):
        """
        Save the params to the params_path.
        Returns the paths to the saved params.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    def load_params(self, param_path):
        """
        Load the params from the param_path.
        Returns the loaded params.
        """    
        raise NotImplementedError("Must be implemented in inheriting class.")

    def save_outputs(self, params, outputs, outputs_path):
        """
        Save the outputs to the outputs_path.
        Returns the paths to the saved outputs.
        """
        raise NotImplementedError("Must be implemented in inheriting class.")

    #-------- Built in --------#
    def generate_pairs(self, nb_params, out_paths, pairs_path, verbose=False, progress_callback=None):
        """
        Generate pairs of simulations to be ranked.
        Add the new simulations to the pairs.csv for all possible pairs, including existing ones.
        
        Args:
            nb_params: Number of parameters to generate
            out_paths: Dictionary containing output paths
            pairs_path: Path to the pairs CSV file
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

        # Run simulations
        report_progress("Running simulations...")
        outputs = self.run(params)

        # Save parameters
        report_progress("Saving parameters...")
        self.save_params(params, out_paths['params'])
        # Save outputs
        report_progress("Saving outputs...")
        self.save_outputs(params, outputs, out_paths['outputs'])

        # Save videos
        report_progress("Generating and saving videos...")
        self.save_videos(params, outputs, out_paths['videos'])

        # Load existing pairs from CSV
        report_progress("Loading existing pairs...")
        if os.path.exists(pairs_path):
            pairs_df = pd.read_csv(pairs_path)
            existing_hashs = set(pairs_df['param1']).union(set(pairs_df['param2']))
        else:
            pairs_df = pd.DataFrame(columns=['param1', 'param2', 'winner'])
            existing_hashs = set()

        # Generate all possible pairs of new simulations
        report_progress("Generating new pairs...")
        hashs = self.generator.hash_params(params)
        new_pairs = list(itertools.combinations(hashs, 2))

        # Generate pairs with existing simulations
        report_progress("Combining with existing simulations...")
        for new_hash in hashs:
            for existing_hash in existing_hashs:
                new_pairs.append((new_hash, existing_hash))

        # Add new pairs to the DataFrame
        report_progress(f"Adding {len(new_pairs)} new pairs...")
        new_pairs_df = pd.DataFrame(new_pairs, columns=['param1', 'param2'])
        new_pairs_df['winner'] = None
        pairs_df = pd.concat([pairs_df, new_pairs_df], ignore_index=True)

        # Save the updated DataFrame back to the CSV
        report_progress("Saving pairs to CSV...")
        pairs_df.to_csv(pairs_path, index=False)
        
        return True

    def save_videos(self, params, outputs, vids_path):
        """
        Generate videos from the outputs.
        Returns the paths to the videos.
        """
        hashs = self.generator.hash_params(params)
        res = []
        for i, output in enumerate(outputs):
            res.append(os.path.join(vids_path, f"{hashs[i]}.mp4"))
            self.save_video_from_output(output, res[-1])
        return res

        
