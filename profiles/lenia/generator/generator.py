import rlhfalife.utils
from profiles.lenia.main_utils import gen_batch_params

class Lenia_Generator(rlhfalife.utils.Generator):
    """
        Generator class for Lenia
    """
    def __init__(self, rewarder):
        super().__init__()
        self.rewarder = rewarder
        
    def generate(self, nb_params): # TODO: Use rewarder as filter (generate 10 times more and filter)
        """
            Generates nb parameters for the simulation.
        """
        p = gen_batch_params(nb_params)

        # transform to list of dictionaries to fit the expected format
        res = []
        for i in range(nb_params):
            res.append(dict(k_size = p['k_size'], mu = p['mu'][i], sigma = p['sigma'][i], beta = p['beta'][i],
                            mu_k = p['mu_k'][i], sigma_k = p['sigma_k'][i], weights = p['weights'][i]))
        return res
    
    def hash_params(self, params):
        """
            Overides the hash_params method because the parameters are dictionaries.
        """
        res = []
        for param in params:
            res.append(hash(str(param)))
        return res

    def train(self, simulator, rewarder):
        """
        Train the generator using the rewarder.
        
        Args:
            simulator: Simulator for which the generator is trained
            rewarder: Rewarder to train with
        """
        self.rewarder = rewarder
        return 
    
    def save(self):
        """
        Save the generator to the path.
        
        Args:
            path: Optional path to save the generator to. If not provided, uses self.save_path.
        """
        return
    
    def load(self):
        return
