import os
import torch
from src.labeler import launch_video_labeler
from src.benchmarker import launch_benchmarker
from src.trainer import launch_training
from src.utils import *

#-------- Parameters --------------------------#
import profiles.lenia.lenia as lenia

# Initialize components
print("Initializing components...")
profile = "lenia"                   # out directory 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Create generator and simulation
print("Initializing rewardor...")
rewardor = lenia.Lenia_Rewardor(device=device)
print("Initializing generator...")
generator = lenia.Lenia_Generator(rewardor)
print("Initializing simulation...")
simulation = lenia.Lenia_Simulation(generator, (3, 400, 400), 0.1, 300)
rewardor.set_simulation(simulation)
#-------- Do not touch below this line --------#

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