import os
import torch
from src.labeler import launch_video_labeler
from src.benchmarker import launch_benchmarker
from src.trainer import launch_training
from src.utils import *
import importlib

#--------------- Profile Selection ---------------#
# List available profiles
profiles = [d for d in os.listdir("profiles") if os.path.isdir(os.path.join("profiles", d))]
print("Available profiles:")
for p in profiles:
    print(f"- {p}")

profile = input("\nPlease enter the profile you want to use: ")
profile_file_path = os.path.join("profiles", profile, f"{profile}.py")

# Load the module dynamically
try:
    spec = importlib.util.spec_from_file_location(profile, profile_file_path)
    profile_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile_module)
except Exception as e:
    print(f"Error loading profile '{profile}': {str(e)}")
    exit(1)

#--------------- Out Paths ---------------#
# Setup the out folders (outputs, videos)
out_path = os.path.join("out", profile)
outputs_path = os.path.join(out_path, "outputs")
videos_path = os.path.join(out_path, "videos")
params_path = os.path.join(out_path, "params")
rewardor_path = os.path.join(out_path, "rewardor")
generator_path = os.path.join(out_path, "generator")

out_paths = {
    'outputs': outputs_path,
    'videos': videos_path,
    'params': params_path,
    'rewardor': rewardor_path,
    'generator': generator_path
}

os.makedirs(out_path, exist_ok=True)
os.makedirs(outputs_path, exist_ok=True)
os.makedirs(videos_path, exist_ok=True)
os.makedirs(params_path, exist_ok=True)
os.makedirs(rewardor_path, exist_ok=True)
os.makedirs(generator_path, exist_ok=True)

# Create the pairs.csv file if it does not exist
pairs_path = os.path.join(out_path, "pairs.csv")
if not os.path.exists(pairs_path):
    with open(pairs_path, "w") as f:
        f.write("param1,param2,winner\n")

#--------------- Loading ---------------#
loader = profile_module.Loader()
generator, rewardor, simulation = loader.load(out_paths)

#-------- Menu System --------#
def print_menu():
    print("\nAlifeHub - Main Menu")
    print("1. Label videos (needs GUI)")
    print("2. Benchmark rewardor (needs GUI)")
    print("3. Launch training")
    print("4. Exit")
    return input("Please choose an option (1-4): ")

def main():
    while True:
        choice = print_menu()
        
        if choice == "1":
            launch_video_labeler(simulation, pairs_path, out_paths, verbose=False)
        elif choice == "2":
            launch_benchmarker(simulation, generator, rewardor, out_paths, verbose=False)
        elif choice == "3":
            launch_training(generator, rewardor, pairs_path, out_paths)
        elif choice == "4":
            print("Exiting AlifeHub...")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()