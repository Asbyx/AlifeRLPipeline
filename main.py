import os
from rlhfalife.labeler import launch_video_labeler
from rlhfalife.benchmarker import launch_benchmarker
from rlhfalife.trainer import launch_training
from rlhfalife.utils import *
from rlhfalife.data_managers import DatasetManager, PairsManager
import json

#--------------- Profile Selection ---------------#
# List available profiles
profiles = [d for d in os.listdir("profiles") if os.path.isdir(os.path.join("profiles", d))]

if len(profiles) == 0:
    print("No profiles found. Please create a profile first. See the README for more information.")
    exit(1)

print("Available profiles:")
for p in profiles:
    if p != "__pycache__":
        print(f"- {p}")
profile = input("\nPlease enter the profile you want to use: ")

while profile not in profiles:
    print(f"Profile '{profile}' not found. Please enter a valid profile.")
    profile = input("\nPlease enter the profile you want to use: ")

# Load the module dynamically
try:
    profile_module = __import__(f"profiles.{profile}", fromlist=[profile])
except Exception as e:
    print(f"Error loading profile '{profile}': {str(e)}")
    exit(1)

#--------------- Config ---------------#
# List available configs
configs = [d for d in os.listdir(os.path.join("profiles", profile, "configs"))]

if len(configs) == 0:
    print("No configs found. Please create configs first. See the README for more information.")
    exit(1)

print("Available configs:")
for c in configs:
    if c != "__pycache__":
        print(f"- {c.split('.')[0]}")
config = input("\nPlease enter the config you want to use: ")

while f"{config}.json" not in configs:
    print(f"Config '{config}' not found. Please enter a valid config.")
    config = input("\nPlease enter the config you want to use: ")

config_file_path = os.path.join("profiles", profile, "configs", f"{config}.json")

config_dict = json.load(open(config_file_path))

#--------------- Out Paths ---------------#
# Setup the out folders (outputs, videos)
out_path = os.path.join("out", profile, config)
outputs_path = os.path.join(out_path, "outputs")
videos_path = os.path.join(out_path, "videos")
params_path = os.path.join(out_path, "params")
rewarder_path = os.path.join(out_path, "rewarder")
generator_path = os.path.join(out_path, "generator")
saved_simulations_path = os.path.join(out_path, "saved_simulations")

out_paths = {
    'outputs': outputs_path,
    'videos': videos_path,
    'params': params_path,
    'rewarder': rewarder_path,
    'generator': generator_path,
    'saved_simulations': saved_simulations_path,
}

os.makedirs(out_path, exist_ok=True)
os.makedirs(outputs_path, exist_ok=True)
os.makedirs(videos_path, exist_ok=True)
os.makedirs(params_path, exist_ok=True)
os.makedirs(rewarder_path, exist_ok=True)
os.makedirs(generator_path, exist_ok=True)
os.makedirs(saved_simulations_path, exist_ok=True)


#--------------- Loading ---------------#
loader = profile_module.Loader()
generator, rewarder, simulator = loader.load(out_paths, config_dict)

#--------------- Data Managers ---------------#
dataset_path = os.path.join(out_path, "dataset.csv")
pairs_path = os.path.join(out_path, "pairs.csv")

dataset_manager = DatasetManager(dataset_path, out_paths, simulator)
pairs_manager = PairsManager(pairs_path)





#-------- Menu System --------#
def print_menu():
    print("\nAlifeHub - Main Menu")
    print("1. Label videos (needs GUI)")
    print("2. Benchmark rewarder (needs GUI)")
    print("3. Launch training")
    print("4. Exit")
    return input("Please choose an option (1-4): ")

def main():
    while True:
        choice = print_menu()
        
        if choice == "1":
            launch_video_labeler(simulator, dataset_manager, pairs_manager, verbose=False)
        elif choice == "2":
            launch_benchmarker(simulator, generator, rewarder, out_paths)
        elif choice == "3":
            launch_training(generator, rewarder, simulator, pairs_manager, dataset_manager)
        elif choice == "4":
            print("Exiting AlifeHub...")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()