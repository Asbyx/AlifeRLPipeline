from .utils import Rewarder, Generator, Simulator
from .data_managers import DatasetManager, PairsManager, TrainingDataset

def launch_training(generator: Generator, rewarder: Rewarder, simulator: Simulator, pairs_manager: PairsManager, dataset_manager: DatasetManager) -> None:
    """
    Launch the training of the rewarder and the generator
    
    Args:
        generator: The generator to train
        rewarder: The rewarder to train
        simulator: The simulator to use
        pairs_manager: PairsManager instance for storing pairs
        dataset_manager: DatasetManager instance for storing simulation data
    """
    print("Creating training dataset...")
    training_dataset = TrainingDataset(pairs_manager, dataset_manager)

    print("Training the rewarder...")
    rewarder.train(training_dataset)

    print("Training the generator...")
    generator.train(simulator, rewarder)
    
    print("Training complete!")
    
    if input("Save the rewarder and generator (Possibility will be given to save them after benchmarking)? (y/n)") == "y":
        rewarder.save()
        generator.save()
        print("Rewarder and generator saved!")

