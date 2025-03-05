from src.utils import Rewardor, Generator

def launch_training(generator: Generator, rewardor: Rewardor, pairs_path, out_paths):
    print("Training the rewardor...")
    rewardor.train(pairs_path, out_paths)
    print("Training the generator...")
    generator.train(rewardor)
    print("Training complete!")
    
    if input("Save the rewardor and generator (Possibility will be given to save them after benchmarkming) ? (y/n)") == "y":
        rewardor.save(out_paths["rewardor"])
        generator.save(out_paths["generator"])

