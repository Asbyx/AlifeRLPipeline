import torch, torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchenhanced import DevModule
from showtens import save_video
from .noise_gen import perlin, perlin_fractal
from .main_utils import gen_batch_params
import src.utils
import pandas as pd
import pickle as pk
import os
import torchvision.models as models
import torch.cuda.amp as amp

class BatchLeniaMC(DevModule):
    """
        Batched Multi-channel lenia, to run batch_size worlds in parallel !
        Does not support live drawing in pygame, mayble will later.
    """
    def __init__(self, size, dt, params=None, state_init = None, device='cpu' ):
        """
            Initializes automaton.  

            Args :
                size : (B,H,W) of ints, size of the automaton and number of batches
                dt : time-step used when computing the evolution of the automaton
                params : dict of tensors containing the parameters. If none, generates randomly
                    keys-values : 
                    'k_size' : odd int, size of kernel used for computations
                    'mu' : (B,3,3) tensor, mean of growth functions
                    'sigma' : (B,3,3) tensor, standard deviation of the growth functions
                    'beta' :  (B,3,3, # of rings) float, max of the kernel rings 
                    'mu_k' : (B,3,3, # of rings) [0,1.], location of the kernel rings
                    'sigma_k' : (B,3,3, # of rings) float, standard deviation of the kernel rings
                    'weights' : (B,3,3) float, weights for the growth weighted sum
                device : str, device 
        """
        super().__init__()
        self.to(device)

        self.batch = size[0]
        self.h, self.w = size[1:]
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        if(params is None):
            params = gen_batch_params(self.batch)
        self.k_size = params['k_size'] # kernel sizes (same for all)

        self.register_buffer('state',torch.rand((self.batch,3,self.h,self.w)))

        if(state_init is None):
            self.set_init_fractal()
        else:
            self.state = state_init.to(self.device)

        self.dt = dt
        # Buffer for all parameters since we do not require_grad for them :

        self.register_buffer('mu', params['mu']) # mean of the growth functions (3,3)
        self.register_buffer('sigma', params['sigma']) # standard deviation of the growths functions (3,3)
        self.register_buffer('beta',params['beta']) # max of the kernel rings (3,3, # of rings)
        self.register_buffer('mu_k',params['mu_k'])# mean of the kernel gaussians (3,3, # of rings)
        self.register_buffer('sigma_k',params['sigma_k'])# standard deviation of the kernel gaussians (3,3, # of rings)
        self.register_buffer('weights',params['weights']) # raw weigths for the growth weighted sum (3,3)

        self.norm_weights()
        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))
        self.kernel = self.compute_kernel() # (3,3,h, w)


    def update_params(self, params):
        """
            Updates the parameters of the automaton. Changes batch size to match one of provided params.
        """
        self.mu = params['mu'] # mean of the growth functions (3,3)
        self.sigma = params['sigma'] # standard deviation of the growths functions (3,3)
        self.beta = params['beta']
        self.mu_k = params['mu_k']
        self.sigma_k = params['sigma_k']
        self.weights = params['weights']
        self.k_size = params['k_size'] # kernel sizes (same for all)
        self.norm_weights()
        self.batch = params['mu'].shape[0] # update batch size

        self.kernel = self.compute_kernel() # (B,3,3,h, w)


    def norm_weights(self):
        # Normalizing the weights
        N = self.weights.sum(dim=1, keepdim = True) # (B,3,3)
        self.weights = torch.where(N > 1.e-6, self.weights/N, 0)

    def get_params(self):
        """
            Get the parameter dictionary which defines the automaton
        """
        params = dict(k_size = self.k_size,mu = self.mu, sigma = self.sigma, beta = self.beta,
                       mu_k = self.mu_k, sigma_k = self.sigma_k, weights = self.weights)
        
        return params

    def set_init_fractal(self):
        """
            Sets the initial state of the automaton using perlin noise
        """
        self.state = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,persistence=0.4) 
    
    def set_init_perlin(self,wavelength=None):
        if(not wavelength):
            wavelength = self.k_size
        self.state = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,black_prop=0.25)
    def kernel_slice(self, r): # r : (k_size,k_size)
        """
            Given a distance matrix r, computes the kernel of the automaton.

            Args :
            r : (k_size,k_size), value of the radius for each location around the center of the kernel
        """

        r = r[None, None, None,None] #(1,1, 1, 1, k_size, k_size)
        r = r.expand(self.batch,3,3,self.mu_k[0][0].size()[0],-1,-1) #(B,3,3,#of rings,k_size,k_size)

        mu_k = self.mu_k[..., None, None] # (B,3,3,#of rings,1,1)
        sigma_k = self.sigma_k[..., None, None]# (B,3,3,#of rings,1,1)

        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) #(B,3,3,#of rings,k_size,k_size)
        #print(K.shape)

        beta = self.beta[..., None, None] # (B,3,3,#of rings,1,1)

        K = torch.sum(beta*K, dim = 3)

        
        return K #(B,3,3,k_size, k_size)
    
    def compute_kernel(self):
        """
            Computes the kernel given the parameters.
        """
        xyrange = torch.arange(-1, 1+0.00001, 2/(self.k_size-1)).to(self.device)
        X,Y = torch.meshgrid(xyrange, xyrange,indexing='ij')
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r) #(B,3,3,k_size,k_size)

        # Normalize the kernel
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,3,3,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(B,3,3,k,k)
    
    def growth(self, u): # u:(B,3,3,H,W)
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (B,3,3,H,W) tensor of concentrations.
        """

        # Possibly in the future add other growth function using bump instead of guassian
        mu = self.mu[..., None, None] # (B,3,3,1,1)
        sigma = self.sigma[...,None,None] # (B,3,3,1,1)
        mu = mu.expand(-1,-1,-1, self.h, self.w) # (B,3,3,H,W)
        sigma = sigma.expand(-1,-1,-1, self.h, self.w) # (B,3,3,H,W)

        return 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1 #(B,3,3,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([self.batch*9,1,self.k_size,self.k_size])#(B*9,1,k,k)

        U = self.state.reshape(1,self.batch*3,self.h,self.w) # (1,B*3,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B*3,H+pad,W+pad)
        
        U = F.conv2d(U, kernel_eff, groups=3*self.batch).squeeze(1) #(B*9,1,H,W) squeeze to (B*9,H,W)
        U = U.reshape(self.batch,3,3,self.h,self.w) # (B,3,3,H,W)

        # assert (self.h,self.w) == (self.state.shape[2], self.state.shape[3])
        
        weights = self.weights[...,None, None] # (B,3,3,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,3,3,H,W)

        dx = (self.growth(U)*weights).sum(dim=1) #(B,3,H,W)

        self.state = torch.clamp(self.state + self.dt*dx, 0, 1)     
    
    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,3) tensor, mass of each channel
        """

        return self.state.mean(dim=(-1,-2)) # (B,3) mean mass for each color

    def draw(self):
        """
            Draws the worldmap from state.
            Separate from step so that we can freeze time,
            but still 'paint' the state and get feedback.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow= self.state[0].permute((2,1,0)) #(W,H,3)

        self._worldmap= toshow.cpu().numpy()   
    
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)
       
class Lenia_Simulation(src.utils.Simulation):
    """
        Simulation class for the Lenia-like automaton.
    """
    def __init__(self, generator, size, dt, run_length, device='cpu'):
        super().__init__(generator)
        self.size = size
        self.dt = dt
        self.run_length = run_length
        self.device = device
        
    def run(self, params):
        """
            Runs the simulation with the given parameters.
            params : list of dict
        """
        # transform the params to expected format for the automaton
        keys = params[0].keys()
        p = {key: torch.stack([torch.tensor(entry[key], device=self.device) for entry in params]) for key in keys}
        p['k_size'] = params[0]['k_size']
        params = p

        automaton = BatchLeniaMC(self.size, self.dt, params, device=self.device)
        B, C, H, W = automaton.state.shape
        outputs = torch.zeros((B, self.run_length, C, H, W), device=self.device)
        for i in range(self.run_length):
            automaton.step()
            outputs[:,i] = 255*automaton.state
        return outputs
    
    def save_video_from_output(self, output, vid_path):
        """
            Converts an output to a video and saves it at vid_path.
        """
        save_video(output, "./", vid_path[:-4])

    def save_output(self, output, output_path):
        """
            Saves the output to the output_path.
            Params must be fully retrievable from the saved file.
        """
        torch.save(output, output_path)

    def save_params(self, params, params_path):
        """
        Save the params to the params_path.
        Returns the paths to the saved params.
        """
        res = []
        hashs = self.generator.hash_params(params)
        for i, param in enumerate(params):
            res.append(os.path.join(params_path, f"{hashs[i]}.pkl"))
            with open(res[-1], "wb") as f:
                pk.dump(param, f)
        return res

    def load_params(self, param_path):
        """
        Load the params from the param_path.
        Returns the loaded params.
        """    
        with open(param_path, "rb") as f:
            return pk.load(f)

    def save_outputs(self, params, outputs, outputs_path):
        """
        Save the outputs to the outputs_path.
        Returns the paths to the saved outputs.
        """
        hashs = self.generator.hash_params(params)
        res = []
        for i, output in enumerate(outputs):
            res.append(os.path.join(outputs_path, f"{hashs[i]}.pkl"))
            self.save_output(output, res[-1])
        return res

    def load_outputs(self, output_paths):
        """
        Load the outputs from the given paths.
        Returns a list of loaded outputs.
        """
        outputs = []
        for path in output_paths:
            with open(path, "rb") as f:
                outputs.append(torch.load(f))
        return outputs

class Lenia_Generator(src.utils.Generator):
    """
        Generator class for the Lenia-like automaton.
    """
    def __init__(self, rewardor):
        super().__init__()
        self.rewardor = rewardor
        
    def generate(self, nb_params): # TODO: Use rewardor as filter (generate 10 times more and filter)
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

    def train(self, rewardor):
        return

class MiniBlock(nn.Module):
    """A simple transformer-style block for processing sequences"""
    def __init__(self, embed_dim, num_tokens, device='cpu'):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.to(device)

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

class CLIPVIPReward(nn.Module):
    """
    A simplified version of CLIPVIP reward model that uses a ResNet backbone.
    Expected video shape: (B, T, C, H, W)
    """
    def __init__(self, num_frames=12, minihead=False, device='cpu'):
        super().__init__()
        
        # Use ResNet18 as backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.backbone_dim = 512  # ResNet18's final feature dimension
        
        if minihead:
            self.head = MiniBlock(self.backbone_dim, num_frames, device)
        else:
            self.head = nn.Identity()
            
        self.final = nn.Linear(self.backbone_dim, 1)
        self.to(device)
        
    def forward(self, x):
        """
        x: (B, T, C, H, W) tensor of video frames
        Returns: (B,) tensor of scores
        """
        B, T, C, H, W = x.shape
        
        # Process each frame through the backbone
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, backbone_dim, 1, 1)
        features = features.view(B, T, self.backbone_dim)  # (B, T, backbone_dim)
        
        # Process sequence
        features = self.head(features)  # (B, T, backbone_dim)
        
        # Take the last token's features
        features = features[:, -1]  # (B, backbone_dim)
        
        # Final scoring
        scores = self.final(features)  # (B, 1)
        return scores.squeeze(1)  # (B,)
    
    def save_weights(self, path):
        """Save model weights"""
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """Load model weights"""
        self.load_state_dict(torch.load(path))

class Lenia_Rewardor(src.utils.Rewardor):
    """
    A Rewardor implementation that uses a simplified CLIPVIP model to rank Lenia simulations.
    """
    def __init__(self, num_frames=12, minihead=True, device='cpu'):
        """
        Initialize the CLIPVIP-based rewardor.
        
        Args:
            num_frames: Number of frames to process
            minihead: Whether to use minihead architecture
            device: Device to run the model on
        """
        super().__init__()
        self.model = CLIPVIPReward(
            num_frames=num_frames,
            minihead=minihead,
            device=device
        )
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.num_frames = num_frames

    def set_simulation(self, simulation):
        self.simulation = simulation
    
    def rank(self, data):
        """
        Rank the data using the CLIPVIP model.
        
        Args:
            data: Tensor of shape (B, T, C, H, W) containing video frames
            
        Returns:
            Tensor of shape (B,) containing scores for each video
        """
        torch.cuda.empty_cache()
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device):
            video1, video2 = data
            # Add batch dimension
            video1 = video1.unsqueeze(0)
            video2 = video2.unsqueeze(0)

            # Ensure videos are in the correct format
            if video1.shape[1] == 3:  # If in format (C,T,H,W)
                video1 = video1.permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)
            if video2.shape[1] == 3:  # If in format (C,T,H,W)
                video2 = video2.permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)

            # Ensure videos are the correct number of frames
            video1 = video1[:, :self.num_frames]
            video2 = video2[:, :self.num_frames]
                
            video1 = video1.to(self.device)
            video2 = video2.to(self.device)
            scores = self.model(video1), self.model(video2)
        return scores

    def train(self, pairs_path, out_path):
        """
        Train the rewardor using paired comparisons.
        
        Args:
            pairs_path: Path to CSV file containing paired comparisons
        """
        pairs_df = pd.read_csv(pairs_path, dtype='str')
        # Filter out rows where winner is null
        pairs_df = pairs_df[pairs_df['winner'].notna()]
        
        print(f"Starting training with {len(pairs_df)} comparison pairs")
        
        # clear cache
        torch.cuda.empty_cache()

        self.model.train()
        total_loss = 0
        correct_predictions = 0
        
        # Convert pairs to training data
        for idx, row in pairs_df.iterrows():
            param1_path = os.path.join(out_path["outputs"], f"{row['param1']}.pkl")
            param2_path = os.path.join(out_path["outputs"], f"{row['param2']}.pkl")
            
            # Load the outputs
            video1 = self.simulation.load_outputs([param1_path])[0]
            video2 = self.simulation.load_outputs([param2_path])[0]

            # Add batch dimension
            video1 = video1.unsqueeze(0)
            video2 = video2.unsqueeze(0)

            # Ensure videos are in the correct format
            if video1.shape[1] == 3:  # If in format (C,T,H,W)
                video1 = video1.permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)
            if video2.shape[1] == 3:  # If in format (C,T,H,W)
                video2 = video2.permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)

            # Ensure videos are the correct number of frames
            video1 = video1[:, :self.num_frames]
            video2 = video2[:, :self.num_frames]
                
            video1 = video1.to(self.device)
            video2 = video2.to(self.device)
            
            # Create labels (1 if param1 won, 0 if param2 won)
            label = torch.tensor([1.0 if row['winner'] == row['param1'] else 0.0]).to(self.device)
            
            # Train step
            self.optimizer.zero_grad()
            score1 = self.model(video1)
            score2 = self.model(video2)
            
            # Compute loss using binary cross entropy
            pred = torch.sigmoid(score1 - score2)
            loss = self.criterion(pred, label)
            
            # Track metrics
            total_loss += loss.item()
            predicted_winner = 1 if pred.item() > 0.5 else 0
            actual_winner = 1 if row['winner'] == row['param1'] else 0
            correct_predictions += (predicted_winner == actual_winner)
            
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            avg_loss = total_loss / (idx + 1)
            accuracy = correct_predictions / (idx + 1) * 100
            if (idx + 1) % 5 == 0:
                print(f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f})")
                print(f"Prediction: {pred.item():.4f} (threshold: 0.5)")
                print(f"Current accuracy: {accuracy:.2f}%")
                print(f"Completed {idx + 1}/{len(pairs_df)} pairs")
                print()
            
        # Print final statistics
        print(f"Final average loss: {total_loss / len(pairs_df):.4f}")
        print(f"Final accuracy: {(correct_predictions / len(pairs_df)) * 100:.2f}%")

    def save(self, path):
        """
        Save the model state to the specified path.
        
        Args:
            path: Path to save the model state
        """
        self.model.save_weights(path+"/rewardor.pth")

    def load(self, path):
        """
        Load the model state from the specified path.
        
        Args:
            path: Path to load the model state from
        """
        self.model.load_weights(path+"/rewardor.pth")

