import os
import numpy as np
from typing import List, Any, Dict
from tqdm import tqdm

def run_genetic_search(
    generator,
    simulator,
    rewarder,
    out_paths: Dict[str, str],
    epsilon: float = 0.01,
    max_steps: int = 100
) -> List[Any]:
    """
    Run a genetic search algorithm using the given generator, simulator, and rewarder.
    
    Args:
        generator: Generator instance that must implement mutation and crossover
        simulator: Simulator instance for running simulations
        rewarder: Rewarder instance for ranking the solutions
        out_paths: Dictionary with output paths
        epsilon: Minimum distance between generations to continue (optional)
        max_steps: Maximum number of steps to run
        
    Returns:
        The best parameters found
    """
    # Check that generator has required methods
    required_methods = ["mutation", "crossover", "distance"]
    for method in required_methods:
        if not hasattr(generator, method) or not callable(getattr(generator, method)):
            raise AttributeError(f"Generator must implement {method} method")
            
    # Verify method signatures by checking their code
    test_param = generator.generate(1)[0]  # Get a sample parameter to test with
    try:
        generator.mutation(test_param)
        generator.crossover(test_param, test_param)
        generator.distance(test_param, test_param)
    except TypeError as e:
        raise AttributeError(f"Generator methods have incorrect signatures: {str(e)}")
    except NotImplementedError:
        pass  # This is fine - means the method is defined but not implemented

    evolution_dir = out_paths["evolution"]

    # Initial population
    population_size = 16
    elite_size = 4
    params = generator.generate(population_size)
    
    # Main evolution loop
    for step in tqdm(range(max_steps)):
        # Run simulations and get rewards
        outputs = simulator.run(params)
        rewards = rewarder.rank(outputs)
        
        # Sort by reward
        sorted_indices = np.argsort(rewards)[::-1]  # Descending order
        sorted_params = [params[i] for i in sorted_indices]
        
        # Select elite
        elite = sorted_params[:elite_size]
        new_params = []
        new_params.extend(elite)
        
        # Generate offspring through crossover
        for i in range(elite_size, population_size):
            # Select two parents from elite
            parent1, parent2 = np.random.choice(elite, 2, replace=False)
            child = generator.crossover(parent1, parent2)
            child = generator.mutation(child)
            new_params.append(child)
        
        # Update population
        params = new_params

        # Check termination criteria
        if has_distance_fn and step > 0:
            mean_distance = np.mean([generator.distance(params[i], params[j]) 
                                   for i in range(population_size) 
                                   for j in range(i+1, population_size)])
            if mean_distance < epsilon:
                print(f"Converged at step {step} with mean distance {mean_distance}")
                break
    
    # Save top parameters
    for i in range(elite_size):
        simulator.save_param(sorted_params[i], os.path.join(evolution_dir, f"best_params_step_{step}"))

    print(f"Converged at step {step} with mean distance {mean_distance}, best score: {rewards[sorted_indices[0]]}.")
    
