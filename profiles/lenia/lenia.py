import os
import cv2
import torch,torch.nn,torch.nn.functional as F
import numpy as np
from torchenhanced import DevModule
from tqdm import tqdm
from showtens import save_video
from .noise_gen import perlin, perlin_fractal
from .main_utils import gen_batch_params
import src.utils

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

        self.batch= size[0]
        self.h, self.w  = size[1:]
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        if(params is None):
            params = gen_batch_params(self.batch,device)
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

class Lenia_Generator(src.utils.Generator):
    """
        Generator class for the Lenia-like automaton.
    """
    def __init__(self):
        super().__init__()
        
    def generate(self, nb):
        """
            Generates nb parameters for the simulation.
        """
        p = gen_batch_params(nb)

        # transform to list of dictionaries to fit the expected format
        res = []
        for i in range(nb):
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

        