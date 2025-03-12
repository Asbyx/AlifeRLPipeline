from src.utils import Rewarder, Generator, Simulation

def launch_training(generator: Generator, rewarder: Rewarder, simulation: Simulation, pairs_path, out_paths):
    print("Training the rewarder...")
    rewarder.train(pairs_path, out_paths)
    print("Training the generator...")
    generator.train(simulation, rewarder)
    print("Training complete!")
    
    if input("Save the rewarder and generator (Possibility will be given to save them after benchmarkming) ? (y/n)") == "y":
        rewarder.save(out_paths["rewarder"])
        generator.save(out_paths["generator"])

