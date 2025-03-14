import os
from rlhfalife.labeler import launch_video_labeler
from rlhfalife.benchmarker import launch_benchmarker
from rlhfalife.trainer import launch_training
from rlhfalife.utils import *
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
    profile_module = __import__(f"profiles.{profile}", fromlist=[profile])
except Exception as e:
    print(f"Error loading profile '{profile}': {str(e)}")
    exit(1)

#--------------- Out Paths ---------------#
# Setup the out folders (outputs, videos)
out_path = os.path.join("out", profile)
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
# Create the pairs.csv file if it does not exist
pairs_path = os.path.join(out_path, "pairs.csv")
if not os.path.exists(pairs_path):
    with open(pairs_path, "w") as f:
        f.write("param1,param2,winner\n")

#--------------- Loading ---------------#
loader = profile_module.Loader()
generator, rewarder, simulator = loader.load(out_paths)

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
            launch_video_labeler(simulator, pairs_path, out_paths, verbose=False)
        elif choice == "2":
            launch_benchmarker(simulator, generator, rewarder, out_paths)
        elif choice == "3":
            launch_training(generator, rewarder, simulator, pairs_path, out_paths)
        elif choice == "4":
            print("Exiting AlifeHub...")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()