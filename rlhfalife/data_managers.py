import os
import pandas as pd
from typing import List, Any, Optional, Tuple, Iterator


class DatasetManager:
    """
    Manages the dataset of simulation parameters, outputs, and videos.
    
    This class handles the storage and retrieval of simulation data, including:
    - Parameters (stored in out/params/)
    - Outputs (stored in out/outputs/)
    - Videos (stored in out/videos/)
    
    It maintains a CSV file (dataset.csv) with columns:
    hash,param_path,output_path,video_path
    """
    
    def __init__(self, dataset_path: str, out_paths: dict, simulator=None):
        """
        Initialize the DatasetManager.
        
        Args:
            dataset_path: Path to the dataset CSV file
            out_paths: Dictionary containing paths to various output directories
            simulator: Optional simulator instance for loading data
        """
        self.dataset_path = dataset_path
        self.out_paths = out_paths
        self.simulator = simulator
        self.data_df = self._load_or_create_dataset()
        
    def _load_or_create_dataset(self) -> pd.DataFrame:
        """Load the dataset from CSV or create a new one if it doesn't exist."""
        if os.path.exists(self.dataset_path):
            return pd.read_csv(self.dataset_path, dtype=str)
        else:
            # Create a new dataset with the required columns
            df = pd.DataFrame(columns=['hash', 'param_path', 'output_path', 'video_path'])
            df.to_csv(self.dataset_path, index=False)
            return df
    
    def save(self):
        """Save the dataset to the CSV file."""
        self.data_df.to_csv(self.dataset_path, index=False)
    
    def reset(self):
        """
        Reset the dataset by deleting all files and clearing the dataframe.
        """
        # Get all file paths
        all_param_files = []
        all_output_files = []
        all_video_files = []
        
        for _, row in self.data_df.iterrows():
            if os.path.exists(row['param_path']):
                all_param_files.append(row['param_path'])
            if os.path.exists(row['output_path']):
                all_output_files.append(row['output_path'])
            if os.path.exists(row['video_path']):
                all_video_files.append(row['video_path'])
                
        # Delete all files
        for file_path in all_param_files + all_output_files + all_video_files:
            try:
                os.remove(file_path)
            except (OSError, FileNotFoundError):
                pass  # Ignore errors if file doesn't exist or can't be deleted
                
        # Reset the dataframe
        self.data_df = pd.DataFrame(columns=['hash', 'param_path', 'output_path', 'video_path'])
        
        # Save the empty dataframe
        self.save()
    
    def add_entry(self, hash_value: str, param_path: str, output_path: str, video_path: str):
        """
        Add a new entry to the dataset.
        
        Args:
            hash_value: Hash value of the parameters
            param_path: Path to the saved parameters
            output_path: Path to the saved outputs
            video_path: Path to the saved video
        """
        # Check if the hash already exists
        if hash_value in self.data_df['hash'].values:
            # Update the existing entry
            self.data_df.loc[self.data_df['hash'] == hash_value, 'param_path'] = param_path
            self.data_df.loc[self.data_df['hash'] == hash_value, 'output_path'] = output_path
            self.data_df.loc[self.data_df['hash'] == hash_value, 'video_path'] = video_path
        else:
            # Add a new entry
            new_entry = pd.DataFrame({
                'hash': [hash_value],
                'param_path': [param_path],
                'output_path': [output_path],
                'video_path': [video_path]
            })
            self.data_df = pd.concat([self.data_df, new_entry], ignore_index=True)
        
        # Save the updated dataset
        self.save()
    
    def get_param_paths(self, hash_values: List[str]) -> List[Optional[str]]:
        """Get the paths to the parameters for a list of hashes."""
        param_paths = []
        for hash_value in hash_values:
            if hash_value in self.data_df['hash'].values:
                param_paths.append(self.data_df.loc[self.data_df['hash'] == hash_value, 'param_path'].iloc[0])
            else:
                raise ValueError(f"Hash {hash_value} not found in dataset")
        return param_paths
    
    def get_output_paths(self, hash_values: List[str]) -> List[Optional[str]]:
        """Get the paths to the outputs for a list of hashes."""
        output_paths = []
        for hash_value in hash_values:
            if hash_value in self.data_df['hash'].values:
                output_paths.append(self.data_df.loc[self.data_df['hash'] == hash_value, 'output_path'].iloc[0])
            else:
                raise ValueError(f"Hash {hash_value} not found in dataset")
        return output_paths
    
    def get_video_paths(self, hash_values: List[str]) -> List[Optional[str]]:
        """Get the paths to the videos for a list of hashes."""
        video_paths = []
        for hash_value in hash_values:
            if hash_value in self.data_df['hash'].values:
                video_paths.append(self.data_df.loc[self.data_df['hash'] == hash_value, 'video_path'].iloc[0])
            else:
                raise ValueError(f"Hash {hash_value} not found in dataset")
        return video_paths
    
    def load_param(self, hash_value: str) -> Any:
        """
        Load the parameters for a given hash.
        
        Args:
            hash_value: Hash value of the parameters to load
            
        Returns:
            The loaded parameters or None if not found
        """
        if not self.simulator:
            raise ValueError("Simulator not provided to DatasetManager. Cannot load parameters.")
            
        param_path = self.get_param_paths([hash_value])[0]
        if param_path:
            return self.simulator.load_param(param_path)
        return None
    
    def load_params(self, hash_values: List[str]) -> List[Any]:
        """
        Load multiple parameters for given hashes.
        
        Args:
            hash_values: List of hash values to load
            
        Returns:
            List of loaded parameters
        """
        return [self.load_param(hash_value) for hash_value in hash_values if self.get_param_paths([hash_value])[0]]
    
    def load_output(self, hash_value: str) -> Any:
        """
        Load the output for a given hash.
        """
        return self.simulator.load_output(self.get_output_paths([hash_value])[0])

    def load_outputs(self, hash_values: List[str]) -> List[Any]:
        """
        Load multiple outputs for given hashes.
        """
        return [self.load_output(hash_value) for hash_value in hash_values if self.get_output_paths([hash_value])[0]]

    def get_all_hashes(self) -> List[str]:
        """Get all hash values in the dataset."""
        return self.data_df['hash'].tolist()
    
    def add_entries_from_simulation(self, hashs: List[str], params: List[Any], 
                                   outputs: List[Any], param_paths: List[str], 
                                   output_paths: List[str], video_paths: List[str]):
        """
        Add multiple entries from a simulation run.
        
        Args:
            hashs: List of hash values
            params: List of parameters
            outputs: List of outputs
            param_paths: List of paths to saved parameters
            output_paths: List of paths to saved outputs
            video_paths: List of paths to saved videos
        """
        for i, hash_value in enumerate(hashs):
            self.add_entry(
                hash_value=str(hash_value),
                param_path=param_paths[i],
                output_path=output_paths[i],
                video_path=video_paths[i]
            )

    def __len__(self) -> int:
        """
        Get the number of entries in the dataset.
        """
        return len(self.data_df)


class PairsManager:
    """
    Manages pairs of simulations for comparison.
    
    This class handles the storage and retrieval of pairs of simulations, including:
    - Unranked pairs (where winner is empty)
    - Ranked pairs (where winner is filled)
    
    It maintains a CSV file (pairs.csv) with columns:
    hash1,hash2,winner
    """
    
    def __init__(self, pairs_path: str):
        """
        Initialize the PairsManager.
        
        Args:
            pairs_path: Path to the pairs CSV file
            dataset_manager: Optional DatasetManager instance for loading data
        """
        self.pairs_path = pairs_path
        self._load_or_create_pairs()
        
    def _load_or_create_pairs(self) -> pd.DataFrame:
        """Load the pairs from CSV or create a new one if it doesn't exist."""
        if os.path.exists(self.pairs_path):
            self.pairs_df = pd.read_csv(self.pairs_path, dtype=str)
        else:
            # Create a new pairs file with the required columns
            self.pairs_df = pd.DataFrame(columns=['hash1', 'hash2', 'winner'])
        self.save()

    def reset(self):
        """
        Reset the pairs manager by clearing all pairs and saving an empty pairs file.
        """
        self.pairs_df = pd.DataFrame(columns=['hash1', 'hash2', 'winner'])
        self.save()
        
    def reset_rankings(self):
        """
        Reset only the rankings by clearing all winners but keeping the pairs.
        This preserves the pairs structure but marks them all as unranked.
        """
        self.pairs_df['winner'] = None
        self.save()

    def _reindex_pairs(self):
        """
        Reorders the unranked pairs such that no hash appears n+1 times until all other hashes
        have appeared n times. This ensures users encounter a diverse range of simulations.
        """
        unranked_pairs = self.pairs_df[self.pairs_df['winner'].isna()].copy()

        # Create a copy to avoid modifying the original
        pairs = unranked_pairs.copy().reset_index(drop=True)
        
        # Initialize the count dictionary for all unique hashes
        all_hashes = pd.concat([pairs['hash1'], pairs['hash2']]).unique()
        hash_counts = {hash_val: 0 for hash_val in all_hashes}
        
        # Initialize the result dataframe
        ordered_pairs = pd.DataFrame(columns=pairs.columns)
        
        # Continue until all pairs are used
        remaining_pairs = pairs.copy()
        
        while not remaining_pairs.empty:
            # Calculate the current score for each pair based on hash counts
            def pair_score(row):
                return hash_counts[row['hash1']] + hash_counts[row['hash2']]
            
            # Find the pair with the lowest score (least seen hashes)
            remaining_pairs['score'] = remaining_pairs.apply(pair_score, axis=1)
            min_score = remaining_pairs['score'].min()
            min_score_pairs = remaining_pairs[remaining_pairs['score'] == min_score]
            
            # If multiple pairs have the same score, choose one randomly
            chosen_idx = min_score_pairs.sample(n=1).index[0]
            chosen_pair = remaining_pairs.loc[chosen_idx]
            
            # Add the chosen pair to the result
            ordered_pairs = pd.concat([ordered_pairs, pd.DataFrame([chosen_pair[pairs.columns]])], ignore_index=True)
            
            # Update the hash counts
            hash_counts[chosen_pair['hash1']] += 1
            hash_counts[chosen_pair['hash2']] += 1
            
            # Remove the chosen pair from remaining pairs
            remaining_pairs = remaining_pairs.drop(chosen_idx)
        
        self.pairs_df = pd.concat([pd.DataFrame(ordered_pairs), self.pairs_df[self.pairs_df['winner'].notna()]], ignore_index=True).reset_index(drop=True)
        self.save()

    def save(self):
        """Save the pairs to the CSV file."""
        self.pairs_df.to_csv(self.pairs_path, index=False)
    
    def _add_pair(self, hash1: str, hash2: str, winner: Optional[str] = None):
        """
        Add a new pair to the dataset.
        
        Args:
            hash1: Hash value of the first simulation
            hash2: Hash value of the second simulation
            winner: Optional hash value of the winner
        """
        # Check if the pair already exists
        existing_pair = self.pairs_df[
            ((self.pairs_df['hash1'] == hash1) & (self.pairs_df['hash2'] == hash2)) |
            ((self.pairs_df['hash1'] == hash2) & (self.pairs_df['hash2'] == hash1))
        ]
        
        if len(existing_pair) > 0:
            # Update the existing pair
            if winner is not None:
                idx = existing_pair.index[0]
                # Ensure the winner matches one of the hashes
                if winner in [hash1, hash2]:
                    self.pairs_df.at[idx, 'winner'] = winner
        else:
            # Add a new pair
            new_pair = pd.DataFrame({
                'hash1': [hash1],
                'hash2': [hash2],
                'winner': [winner]
            })
            self.pairs_df = pd.concat([self.pairs_df, new_pair], ignore_index=True)
        
        # Save the updated pairs
        self.save()
    
    def add_pairs(self, pairs: List[Tuple[str, str]], winners: Optional[List[str]] = None):
        """
        Add multiple pairs to the dataset.
        
        Args:
            pairs: List of (hash1, hash2) tuples
            winners: Optional list of winner hash values
        """
        if winners is None:
            winners = [None] * len(pairs)
        
        for i, (hash1, hash2) in enumerate(pairs):
            self._add_pair(hash1, hash2, winners[i] if i < len(winners) else None)
        self._reindex_pairs()
    
    def _get_unranked_pairs(self) -> pd.DataFrame:
        """Get all unranked pairs (where winner is null)."""
        return self.pairs_df[self.pairs_df['winner'].isnull()].reset_index(drop=True).copy()
    
    def _get_ranked_pairs(self) -> pd.DataFrame:
        """Get all ranked pairs (where winner is not null)."""
        return self.pairs_df[self.pairs_df['winner'].notnull()].reset_index(drop=True).copy()
    
    def set_winner(self, hash1: str, hash2: str, winner: int):
        """
        Set the winner for a pair.
        
        Args:
            hash1: Hash value of the first simulation
            hash2: Hash value of the second simulation
            winner: 0 or 1 (0 for hash1, 1 for hash2)
        """
        if winner not in [0, 1]:
            raise ValueError(f"Winner ({winner}) must be either 0 or 1")
        
        # Find the pair
        pair_idx = self.pairs_df[
            ((self.pairs_df['hash1'] == hash1) & (self.pairs_df['hash2'] == hash2)) |
            ((self.pairs_df['hash1'] == hash2) & (self.pairs_df['hash2'] == hash1))
        ].index
        
        if len(pair_idx) > 0:
            self.pairs_df.at[pair_idx[0], 'winner'] = winner if self.pairs_df.at[pair_idx[0], 'hash1'] == hash1 else 1 - winner
        else:
            self._add_pair(hash1, hash2, winner)
    
    def _get_all_pairs(self) -> pd.DataFrame:
        """Get all pairs."""
        return self.pairs_df.copy()

    def get_nb_unranked_pairs(self) -> int:
        """Get the number of unranked pairs."""
        return self._get_unranked_pairs().shape[0]
    
    def get_nb_ranked_pairs(self) -> int:
        """Get the number of ranked pairs."""
        return self._get_ranked_pairs().shape[0]
    
    def get_nb_pairs(self) -> int:
        """Get the total number of pairs."""
        return self.pairs_df.shape[0]

    def get_next_unranked_pair(self) -> Tuple[str, str]:
        """Get the next unranked pair."""
        unranked_pairs = self._get_unranked_pairs()
        if unranked_pairs.empty:
            return None, None
        return unranked_pairs.iloc[0]['hash1'], unranked_pairs.iloc[0]['hash2']
    
    def get_last_ranked_pair(self) -> Tuple[str, str]:
        """Get the last ranked pair."""
        ranked_pairs = self._get_ranked_pairs()
        if ranked_pairs.empty:
            return None, None
        return ranked_pairs.iloc[-1]['hash1'], ranked_pairs.iloc[-1]['hash2']

class TrainingDataset:
    """
    Manages the data for the training of the rewarder
    """
    def __init__(self, pairs_manager: PairsManager, dataset_manager: DatasetManager):
        """
        Initialize the TrainingDataset.
        This dataset delivers triplets of (path_to_output_1, path_to_output_2, winner). The pairs are all ranked. The load of the outputs is expected to be done in the Rewarder, from the paths given by the TrainingDataset.

        Args:
            pairs_manager: The pairs manager
            dataset_manager: The dataset manager
        """        
        ranked_pairs = pairs_manager._get_ranked_pairs()
        
        # Prepare the data
        self.data = []
        simulations = set()
        for _, row in ranked_pairs.iterrows():
            hash1 = row['hash1']
            hash2 = row['hash2']
            winner = 0 if row['winner'] == hash1 else 1
            simulations.add(hash1)
            simulations.add(hash2)
            
            # Get the output paths
            output_path1 = dataset_manager.get_output_paths([hash1])[0]
            output_path2 = dataset_manager.get_output_paths([hash2])[0]
            
            self.data.append((output_path1, output_path2, winner))
        self.simulations_number = len(simulations)

    def __len__(self) -> int:
        """
        Get the number of pairs in the dataset
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        """
        Get the pair at the given index.

        Returns:
            A tuple containing the path_to_output_1, path_to_output_2, {0, 1} (winner is 0 for left, 1 for right)
        """
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of bounds")
        return self.data[index]

    def __iter__(self) -> Iterator[Tuple[str, str, int]]:
        """
        Get an iterator over the pairs

        Returns:
            An iterator over the pairs
        """
        return iter(self.data)
        
