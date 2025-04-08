import os
import json
import argparse
from rlhfalife.labeler import launch_video_labeler
from rlhfalife.benchmarker import launch_benchmarker
from rlhfalife.trainer import launch_training
from rlhfalife.utils import *
from rlhfalife.data_managers import DatasetManager, PairsManager

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
    }

    # Create all directories
    for path in out_paths.values():
        os.makedirs(path, exist_ok=True)

    return out_path, out_paths

def print_menu():
    print("\nAlifeHub - Main Menu")
    print("1. Label videos (needs GUI)")
    print("2. Benchmark rewarder (needs GUI)")
    print("3. Launch training")
    print("4. Change frame size")
    print("5. Reload models and data managers")
    print("0. Exit")
    return input("Please choose an option (0-5): ")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AlifeHub CLI')
    parser.add_argument('--profile', type=str, help='Profile name to use')
    parser.add_argument('--config', type=str, help='Config name to use')
    parser.add_argument('--frame_size', type=int, default=300, help='Frame size to use')
    args = parser.parse_args()

    # Select profile and config
    profile = select_profile(args.profile)
    config = select_config(profile, args.config)

    print(f"Using profile: {profile}, config: {config}, frame size: {args.frame_size}.")


    # Load the profile module
    print(f"Loading profile...")
    try:
        profile_module = __import__(f"profiles.{profile}", fromlist=[profile])
    except Exception as e:
        print(f"Error loading profile '{profile}': {str(e)}")
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
        
        if choice == "1":
            launch_video_labeler(simulator, dataset_manager, pairs_manager, verbose=False, frame_size=(args.frame_size, args.frame_size))
        elif choice == "2":
            launch_benchmarker(simulator, generator, rewarder, out_paths, frame_size=(args.frame_size, args.frame_size))
        elif choice == "3":
            launch_training(generator, rewarder, simulator, pairs_manager, dataset_manager)
        elif choice == "4":
            try:
                new_size = int(input("Enter new frame size (default: 300): ") or "300")
                if new_size > 0:
                    args.frame_size = new_size
                    print(f"Frame size updated to {new_size}")
                else:
                    print("Frame size must be positive")
            except ValueError:
                print("Please enter a valid number")
        elif choice == "5":
            print("Reloading models and data managers.")
        elif choice == "0":
            print("Exiting AlifeHub...")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()