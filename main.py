import os
import tkinter as tk
from src.labeler.labeler import VideoLabelerApp, launch_video_labeler
from src.utils import *

#-------- Parameters --------------------------#
import profiles.lenia.lenia as lenia

profile = "lenia"                   # out directory 
generator = lenia.Lenia_Generator()
simulation = lenia.Lenia_Simulation(generator, (3, 400, 400), 0.1, 300, device='cuda:0')
rewardor = Rewardor()
#-------- Do not touch below this line --------#

# Setup the out folders (outputs, videos)
out_path = os.path.join("out", profile)
outputs_path = os.path.join(out_path, "outputs")
videos_path = os.path.join(out_path, "videos")
params_path = os.path.join(out_path, "params")

out_paths = {
    'outputs': outputs_path,
    'videos': videos_path,
    'params': params_path
}

os.makedirs(out_path, exist_ok=True)
os.makedirs(outputs_path, exist_ok=True)
os.makedirs(videos_path, exist_ok=True)
os.makedirs(params_path, exist_ok=True)

# Create the pairs.csv file if it does not exist
pairs_path = os.path.join(out_path, "pairs.csv")
if not os.path.exists(pairs_path):
    with open(pairs_path, "w") as f:
        f.write("param1,param2,winner\n")

#-------- Launch Video Labeler App --------#
launch_video_labeler(simulation, pairs_path, out_paths, verbose=False)
print('Labeler launched')