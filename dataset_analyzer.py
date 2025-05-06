import os
import pandas as pd
from rlhfalife.data_managers import TrainingDataset, DatasetManager, PairsManager
from collections import Counter
import argparse
import importlib

def analyze_training_dataset(profile, config, simulator=None):
    """
    Analyze a TrainingDataset to get statistics about the rankings.
    
    Args:
        profile: Profile name
        config: Config name
        simulator: Optional simulator object (needed for DatasetManager)
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Setup paths
    out_path = os.path.join("out", profile, config)
    out_paths = {
        'outputs': os.path.join(out_path, "outputs"),
        'videos': os.path.join(out_path, "videos"),
        'params': os.path.join(out_path, "params"),
        'rewarder': os.path.join(out_path, "rewarder"),
        'generator': os.path.join(out_path, "generator"),
        'saved_simulations': os.path.join(out_path, "saved_simulations"),
        'benchmark': os.path.join("out", profile, "benchmark"),
    }
    
    dataset_path = os.path.join(out_path, "dataset.csv")
    pairs_path = os.path.join(out_path, "pairs.csv")
    
    # Create the data managers
    dataset_manager = DatasetManager(dataset_path, out_paths, simulator)
    pairs_manager = PairsManager(pairs_path)
    
    # Create the training dataset
    training_dataset = TrainingDataset(pairs_manager, dataset_manager)
    
    # Get the total number of hashes
    total_hashes = len(dataset_manager.get_all_hashes())
    
    # Get the ranked pairs
    ranked_pairs = pairs_manager._get_ranked_pairs()
    
    # Get all unique hashes from the ranked pairs
    ranked_hashes = set()
    hash_rankings = Counter()
    
    for _, row in ranked_pairs.iterrows():
        hash1 = row['hash1']
        hash2 = row['hash2']
        ranked_hashes.add(hash1)
        ranked_hashes.add(hash2)
        hash_rankings[hash1] += 1
        hash_rankings[hash2] += 1
    
    # Count number of ranked hashes
    num_ranked_hashes = len(ranked_hashes)
    
    # Prepare the hash rankings sorted by count
    hash_ranking_counts = [(h, c) for h, c in hash_rankings.items()]
    hash_ranking_counts.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'total_hashes': total_hashes,
        'num_ranked_hashes': num_ranked_hashes,
        'hash_ranking_counts': hash_ranking_counts,
        'training_dataset_size': len(training_dataset),
        'simulations_number': training_dataset.simulations_number
    }

def print_analysis(analysis):
    """
    Print the analysis results in a readable format.
    
    Args:
        analysis: Analysis results dictionary
    """
    print("\n===== Training Dataset Analysis =====")
    print(f"Total Hashes: {analysis['total_hashes']}")
    print(f"Number of Ranked Hashes: {analysis['num_ranked_hashes']}")
    print(f"Training Dataset Size (ranked pairs): {analysis['training_dataset_size']}")
    print(f"Simulations in Training Dataset: {analysis['simulations_number']}")
    
    # Print ranking distribution statistics
    counts = [count for _, count in analysis['hash_ranking_counts']]
    if counts:
        print("\n----- Ranking Distribution Statistics -----")
        print(f"Max Rankings per Hash: {max(counts)}")
        print(f"Min Rankings per Hash: {min(counts)}")
        print(f"Average Rankings per Hash: {sum(counts) / len(counts):.2f}")
        
        # Calculate median
        sorted_counts = sorted(counts)
        mid = len(sorted_counts) // 2
        if len(sorted_counts) % 2 == 0:
            median = (sorted_counts[mid-1] + sorted_counts[mid]) / 2
        else:
            median = sorted_counts[mid]
        print(f"Median Rankings per Hash: {median:.1f}")
        
        # Calculate distribution by range
        ranges = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
        print("\n----- Ranking Count Distribution -----")
        for start, end in ranges:
            end_str = f"{end}" if end != float('inf') else "+"
            count_in_range = sum(1 for c in counts if start <= c <= end)
            percentage = (count_in_range / len(counts)) * 100
            print(f"{start}-{end_str}: {count_in_range} hashes ({percentage:.1f}%)")
    
    print("\n----- Top 10 Most Ranked Hashes -----")
    for i, (hash_val, count) in enumerate(analysis['hash_ranking_counts'][:10]):
        print(f"{i+1}. Hash: {hash_val[:8]}... - Rankings: {count}")
    
    print("\n----- Least Ranked Hashes (Bottom 10) -----")
    for i, (hash_val, count) in enumerate(reversed(analysis['hash_ranking_counts'][-10:])):
        print(f"{i+1}. Hash: {hash_val[:8]}... - Rankings: {count}")
    
    # Calculate imbalance score - how uneven the ranking distribution is
    if counts:
        avg = sum(counts) / len(counts)
        imbalance = sum(abs(c - avg) for c in counts) / (len(counts) * avg) if avg > 0 else 0
        print(f"\nRanking Imbalance Score: {imbalance:.2f} (0=perfectly balanced, higher=more imbalanced)")
        
        # Overall summary
        print("\n----- Overall Summary -----")
        if imbalance < 0.3:
            balance_status = "Well balanced"
        elif imbalance < 0.6:
            balance_status = "Moderately balanced"
        else:
            balance_status = "Highly imbalanced"
            
        coverage = analysis['num_ranked_hashes'] / analysis['total_hashes'] * 100 if analysis['total_hashes'] > 0 else 0
        
        if coverage < 50:
            coverage_status = "Low coverage"
        elif coverage < 80:
            coverage_status = "Medium coverage"
        else:
            coverage_status = "High coverage"
            
        print(f"Hash Coverage: {coverage:.1f}% ({coverage_status})")
        print(f"Ranking Balance: {balance_status}")
        
        if coverage < 50 or imbalance > 0.6:
            print("\nRecommendation: Consider generating more diverse pairs for better training data distribution.")
        elif coverage < 80 or imbalance > 0.3:
            print("\nRecommendation: Dataset is reasonably balanced but could be improved with more samples.")
        else:
            print("\nRecommendation: Dataset is well-balanced and has good coverage. Suitable for training.")

def analyze_existing_dataset(dataset_manager, pairs_manager):
    """
    Analyze an existing dataset using already created managers.
    
    Args:
        dataset_manager: The DatasetManager instance
        pairs_manager: The PairsManager instance
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Create the training dataset
    training_dataset = TrainingDataset(pairs_manager, dataset_manager)
    
    # Get the total number of hashes
    total_hashes = len(dataset_manager.get_all_hashes())
    
    # Get the ranked pairs
    ranked_pairs = pairs_manager._get_ranked_pairs()
    
    # Get all unique hashes from the ranked pairs
    ranked_hashes = set()
    hash_rankings = Counter()
    
    for _, row in ranked_pairs.iterrows():
        hash1 = row['hash1']
        hash2 = row['hash2']
        ranked_hashes.add(hash1)
        ranked_hashes.add(hash2)
        hash_rankings[hash1] += 1
        hash_rankings[hash2] += 1
    
    # Count number of ranked hashes
    num_ranked_hashes = len(ranked_hashes)
    
    # Prepare the hash rankings sorted by count
    hash_ranking_counts = [(h, c) for h, c in hash_rankings.items()]
    hash_ranking_counts.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'total_hashes': total_hashes,
        'num_ranked_hashes': num_ranked_hashes,
        'hash_ranking_counts': hash_ranking_counts,
        'training_dataset_size': len(training_dataset),
        'simulations_number': training_dataset.simulations_number
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze a TrainingDataset')
    parser.add_argument('--profile', '-p', required=True, help='Profile name')
    parser.add_argument('--config', '-c', required=True, help='Config name')
    args = parser.parse_args()
    
    # Load profile module (similar to main.py)
    try:
        profile_module = importlib.import_module(f"profiles.{args.profile}")
        print(f"Loaded profile module: {args.profile}")
        
        # Load the profile's loader
        loader = profile_module.Loader()
        
        # Load simulator and other components
        out_path = os.path.join("out", args.profile, args.config)
        out_paths = {
            'outputs': os.path.join(out_path, "outputs"),
            'videos': os.path.join(out_path, "videos"),
            'params': os.path.join(out_path, "params"),
            'rewarder': os.path.join(out_path, "rewarder"),
            'generator': os.path.join(out_path, "generator"),
            'saved_simulations': os.path.join(out_path, "saved_simulations"),
            'benchmark': os.path.join("out", args.profile, "benchmark"),
        }
        
        # Load config
        import json
        config_file_path = os.path.join("profiles", args.profile, "configs", f"{args.config}.json")
        config_dict = json.load(open(config_file_path))
        
        _, _, simulator = loader.load(out_paths, config_dict)
        
        # Analyze with simulator
        analysis = analyze_training_dataset(args.profile, args.config, simulator)
        print_analysis(analysis)
        
    except Exception as e:
        # Fallback to analysis without simulator (may not work for all cases)
        print(f"Error loading profile module: {e}")
        print("Attempting to analyze without simulator...")
        try:
            analysis = analyze_training_dataset(args.profile, args.config)
            print_analysis(analysis)
        except Exception as e2:
            print(f"Error analyzing dataset: {e2}")
            print("Try running this script from the main directory of the project.")

if __name__ == "__main__":
    main() 