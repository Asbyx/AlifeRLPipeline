from rlhfalife.utils import Rewarder, Generator, Simulator

def launch_training(generator: Generator, rewarder: Rewarder, simulator: Simulator, pairs_path: str, out_paths: dict) -> None:
    """
    Launch the training of the rewarder and the generator
    
    Args:
        generator: The generator to train
        rewarder: The rewarder to train
        simulator: The simulator to use
        pairs_path: The path to the pairs
        out_paths: The paths to the outputs
    """
    print("Training the rewarder...")
    rewarder.train(pairs_path, out_paths)
    print("Training the generator...")
    generator.train(simulator, rewarder)
    print("Training complete!")
    
    if input("Save the rewarder and generator (Possibility will be given to save them after benchmarkming) ? (y/n)") == "y":
        rewarder.save(out_paths["rewarder"])
        generator.save(out_paths["generator"])
        print("Rewarder and generator saved!")

