# Alife RL pipeline _WIP_
The idea of the app is to be a pipeline applying RLHF to train a model for generating interesting simulation parameters.

The pipeline is a loop of 4 stages:
- **Generation**: A Generator produces some parameters
- **Labeling**: A human makes a ranking of the output of the simulations
- **Training**: A Rewardor is trained to approximate the ranking of a human
- **Fine Tuning**: Using the Rewardor, train the Generator.
- and repeat...

This pipeline include every steps.  
**In order to use it:** one must implement a `Simulation`, a `Generator` and a `Rewardor`, in `src.profiles/<new profile>`, using the abstract classes given in `src.utils.py`.  
**The only requirement** is to have a `Loader` (abstract still available in `src.utils.py`) in a file `<new profile>.py`, the rest is free to implement as your convenience!
Then, by running the main, the full pipeline is launched and the training starts.

# WIP
- Labeler: 
    - put the window at front when launching
    - display rewardor scores
    - display % of labeling