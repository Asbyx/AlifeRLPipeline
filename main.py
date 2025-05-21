import os
import json
import argparse
import importlib
import sys
import shutil
from rlhfalife.labeler import launch_video_labeler
from rlhfalife.quad_labeler import launch_quad_labeler
from rlhfalife.benchmarker import launch_benchmarker
from rlhfalife.trainer import launch_training
from rlhfalife.utils import *
from rlhfalife.data_managers import DatasetManager, PairsManager
from exporter import export_profile_interactive
from dataset_analyzer import analyze_existing_dataset, print_analysis

def get_available_profiles():
    """Get list of available profiles."""
    return [d for d in os.listdir("profiles") if os.path.isdir(os.path.join("profiles", d)) and d != "__pycache__"]

def get_available_configs(profile):
    """Get list of available configs for a profile."""
    return [c.split('.')[0] for c in os.listdir(os.path.join("profiles", profile, "configs")) 
            if c != "__pycache__" and c.endswith('.json')]

def select_profile(profile=None):
    """Select a profile, either from argument or via prompt."""
    profiles = get_available_profiles()
    
    if len(profiles) == 0:
        print("No profiles found. Please create a profile first. See the README for more information.")
        exit(1)

    if profile is None:
        print("Available profiles:")
        for p in profiles:
            print(f"- {p}")
        profile = input("\nPlease enter the profile you want to use: ")

        while profile not in profiles:
            print(f"Profile '{profile}' not found. Please enter a valid profile.")
            profile = input("\nPlease enter the profile you want to use: ")
    elif profile not in profiles:
        print(f"Error: Profile '{profile}' not found.")
        exit(1)

    return profile

def select_config(profile, config=None):
    """Select a config, either from argument or via prompt."""
    configs = get_available_configs(profile)
    
    if len(configs) == 0:
        print("No configs found. Please create configs first. See the README for more information.")
        exit(1)

    if config is None:
        print("Available configs:")
        for c in configs:
            print(f"- {c}")
        config = input("\nPlease enter the config you want to use: ")

        while config not in configs:
            print(f"Config '{config}' not found. Please enter a valid config.")
            config = input("\nPlease enter the config you want to use: ")
    elif config not in configs:
        print(f"Error: Config '{config}' not found in profile '{profile}'.")
        exit(1)

    return config

def setup_paths(profile, config):
    """Setup and return all necessary paths."""
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

    # Create all directories
    for path in out_paths.values():
        os.makedirs(path, exist_ok=True)

    return out_path, out_paths

def load_profile_module(profile):
    """Load the profile module and return it, forcing a reload if it was previously loaded."""
    module_name = f"profiles.{profile}"
    try:
        if module_name in sys.modules:
            submodules = []
            # First identify all submodules that need to be reloaded
            for name in list(sys.modules.keys()):
                if name.startswith(f"{module_name}."):
                    submodules.append(name)
            
            # Reload all submodules
            for submodule in submodules:
                if submodule in sys.modules:
                    importlib.reload(sys.modules[submodule])
                    
            profile_module = importlib.reload(sys.modules[module_name])
        else:
            profile_module = importlib.import_module(module_name)
        return profile_module
    except Exception as e:
        print(f"Error loading profile '{profile}': {str(e)}")
        return None

def generate_pairs_cli(simulator, dataset_manager, pairs_manager, num_sims, verbose=True):
    """
    Generate pairs without GUI.
    
    Args:
        simulator: Simulator object to use for generating pairs
        dataset_manager: DatasetManager instance for storing simulation data
        pairs_manager: PairsManager instance for storing pairs
        num_sims: Number of simulations to generate
        verbose: Whether to print verbose output
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:            
        print(f"Generating {num_sims} simulations...")
        simulator.generate_pairs(
            num_sims, 
            dataset_manager, 
            pairs_manager, 
            verbose=verbose
        )
        print(f"Successfully generated {num_sims} simulations.")
        return True
    except Exception as e:
        print(f"Error generating pairs: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def print_menu():
    print("\nAlifeHub - Main Menu")
    print("1. Label Pairs (needs GUI)")
    print("2. Quad Labeler (needs GUI)")
    print("3. Benchmark rewarder")
    print("4. Launch training")
    print("5. Generate pairs (no GUI)")
    print("\n6. Reload new code")
    print("7. Change frame size")
    print("8. Reload models and data managers (updates config)")
    print("9. Export profile")
    print("A. Analyze training dataset")
    print("B. Reset labels (keep simulations and pairs)")
    print("C. Reset config (erase everything)")
    print("0. Exit")
    return input("Please choose an option (0-9, A-C): ")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AlifeHub CLI')
    parser.add_argument('--profile', '-p', type=str, help='Profile name to use')
    parser.add_argument('--config', '-c', type=str, help='Config name to use')
    parser.add_argument('--frame_size', '-f', type=int, default=300, help='Frame size to use')
    args = parser.parse_args()

    # Select profile and config
    profile = select_profile(args.profile)
    config = select_config(profile, args.config)

    print(f"Using profile: {profile}, config: {config}, frame size: {args.frame_size}.")

    # Load the profile module
    print(f"Loading profile...")
    profile_module = load_profile_module(profile)
    if profile_module is None:
        exit(1)

    # Load the config
    config_file_path = os.path.join("profiles", profile, "configs", f"{config}.json")
    config_dict = json.load(open(config_file_path))

    # Setup paths
    out_path, out_paths = setup_paths(profile, config)
    dataset_path = os.path.join(out_path, "dataset.csv")
    pairs_path = os.path.join(out_path, "pairs.csv")

    # Main loop
    loader = profile_module.Loader()
    
    while True:
        print("Loading the generator, rewarder and simulator...")
        generator, rewarder, simulator = loader.load(out_paths, config_dict)
        
        print("\nBuilding the data managers...")
        dataset_manager = DatasetManager(dataset_path, out_paths, simulator)
        pairs_manager = PairsManager(pairs_path)
        print(f"Number of simulations: {len(dataset_manager)}")
        print(f"Number of ranked pairs: {pairs_manager.get_nb_ranked_pairs()}")
        
        choice = print_menu()
        print()
        
        match choice:
            case "1":
                launch_video_labeler(simulator, dataset_manager, pairs_manager, verbose=False, frame_size=(args.frame_size, args.frame_size))
            case "2":
                launch_quad_labeler(simulator, dataset_manager, pairs_manager, verbose=False, frame_size=(args.frame_size, args.frame_size))
            case "3":
                launch_benchmarker(simulator, generator, rewarder, out_paths, frame_size=(args.frame_size, args.frame_size))
            case "4":
                launch_training(generator, rewarder, simulator, pairs_manager, dataset_manager)
            case "5":
                try:
                    num_sims = int(input("Enter number of simulations to generate: ") or "5")
                    if num_sims <= 0:
                        print("Number of simulations must be positive")
                        continue
                    
                    generate_pairs_cli(simulator, dataset_manager, pairs_manager, num_sims)
                except ValueError:
                    print("Please enter a valid number")
            case "6":
                print("Reloading new code...")            
                profile_module = load_profile_module(profile)
                if profile_module is None:
                    exit(1)
                loader = profile_module.Loader()
            case "7":
                try:
                    new_size = int(input("Enter new frame size (default: 300): ") or "300")
                    if new_size > 0:
                        args.frame_size = new_size
                        print(f"Frame size updated to {new_size}")
                    else:
                        print("Frame size must be positive")
                except ValueError:
                    print("Please enter a valid number")
            case "8":
                print("Reloading models and data managers...")
                config_file_path = os.path.join("profiles", profile, "configs", f"{config}.json")
                config_dict = json.load(open(config_file_path))
            case "9":
                print("\nExport Profile")
                export_profile_interactive()
            case "A" | "a":
                print("\nAnalyzing Training Dataset")
                analysis = analyze_existing_dataset(dataset_manager, pairs_manager)
                print_analysis(analysis)
                input("\nPress Enter to continue...")
            case "B" | "b":
                print("Resetting Labels ")
                confirmation = input("Are you sure you want to reset all labels? This will clear all winners. (y/n): ")
                if confirmation.lower() == 'y':
                    pairs_manager.reset_rankings()
                    print(f"All rankings have been reset. Number of unranked pairs: {pairs_manager.get_nb_unranked_pairs()}")
                else:
                    print("Reset operation cancelled.")
            case "C" | "c":
                print("Resetting Config")
                confirmation = input(f"Are you sure you want to reset the config '{config}' for profile '{profile}'? This will delete all associated data. (y/n): ")
                if confirmation.lower() == 'y':
                    try:
                        shutil.rmtree(out_path)
                        print(f"Config '{config}' for profile '{profile}' has been reset.")
                        print("Please restart the application.")
                        exit(0)
                    except Exception as e:
                        print(f"Error resetting config: {str(e)}")
                else:
                    print("Reset operation cancelled.")
            case "0":
                print("Exiting AlifeHub...")
                break
            case _:
                print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()