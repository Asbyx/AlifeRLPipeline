# Alife RLH pipeline _WIP_
The idea of the app is to be a pipeline applying RLHF to train a model for generating interesting simulation parameters.

The pipeline is a loop of 4 stages:
- **Generation**: A Generator produces some parameters
- **Labeling**: A human makes a ranking of the output of the simulations
- **Training**: A Rewarder is trained to approximate the ranking of a human
- **Fine Tuning**: Using the Rewarder, train the Generator.
- and repeat...
This pipeline include every steps.  

## Installation
Run `pip install -e .` in order to install the required packages and the `rlhfalife` package which you can easily import in any files to use. It is not expected to import anything else than the `rlhfalife.utils` module

## Usage
One must implement a python package, containing a `Simulator`, a `Generator` and a `Rewarder`, in `profiles/<new profile>`, using the abstract classes given in `rlhfalife.utils.py` (it is recommended to inherit the classes).   
**The only requirement** is to have a `Loader` (abstract still available in `rlhfalife.utils.py`), which is made available through an `__init__.py` file in the package. The rest is free to implement as your convenience!

An example is provided with the `lenia` profile. Note that the files structure is not mandatory, only the `__init__.py` should be present.

Then, by running the main, the full pipeline is launched and the training starts.  
The pipeline is divided in three categories:
- **Label videos**: A window with 2 simulations will be presented, the user must select the "most interesting" one. Videos generation is fully automated.
- **Benchmark rewarder**: 10 simulations will be generated and presented to the user, along with their scores given by the rewarder. That way, the user can check that the rewarder is indeed well aligned with the expectations.
- **Launch training**: Simply launch a rewardor and then a generator training, based on the labeled simulations.

# WIP
- Integrate feedback
- Labeler: 
    - Propose to wipe existing pairs and regenerate an arbitrary number of pairs
- Benchmarker:
    - Weird behavior uppon reading some videos
    - Propose to save the models or not
- Correct the requirements.txt
- Profile downloading and sharing
- Propose to reload everything (after modification for example)