import torch.nn as nn, torch
from torch import Tensor
import random
from torchenhanced import DevModule
from collections.abc import Callable


class ArbitraryFunction(nn.Module):
    """
    Batched Arbitrary Function Generator. It is agnostic to a specific shape of input,
    just expect a tensor of shape (N,*), where N is a 'batch size', and * is the rest of the dimensions,
    where the function is evaluated in. All arbitrary function of the batch will have the same number of harmonics.
    """

    def __init__(
        self,
        coefficients: Tensor,
        ranges: Tensor,
        harmonics: Tensor = None,
        rescale: tuple = None,
        clips_min=None,
        cut_off=False,
        device="cpu",
    ):
        """
        Note that harmonics also include the zeroth harmonic, which is the constant term.

        Args:
            coefficients: (func_num, num_harmonics*2) tensor of coefficients. Will be reshaped
                to (func_num, num_harmonics,2), where the last dimension is (cos,sin) coefficients.
                This is done for compatibility with the complex version.
            ranges: (func_num,2) tensor of function x ranges. E.g., if a funciton is defined in [-1,1], the range is (-1,1)
            harmonics: (func_num, num_harmonics) tensor of harmonics to use. If none, assumes harmonics are the first n_coeffs,
                i.e. [0,1,2,...,n_coeffs-1].
            rescale: tuple, (min,max). Will rescale the function values to this range
            clips_min: float, after rescaling will clip to this value
            cut_off: bool, if True, will set the function to 0 outside the range defined by ranges
            device : torch device
        """
        super().__init__()
        self.device = device

        assert coefficients.ndim == 2, "Coefficients must be a 2D tensor"
        coefficients = coefficients.reshape(
            coefficients.shape[0], -1, 2
        )  # (func_num, num_harmonics,2)

        func_num,num_harmonics,_ = coefficients.shape
        if harmonics is None:
            harmonics = torch.arange(num_harmonics, device=device).float()[None,:].expand(func_num,-1) # (func_num, num_harmonics,)

        assert coefficients[:, :, 0].shape == harmonics.shape, (
            "Coefficients and harmonics must have matchin first two dimensions"
        )
        self.func_num = harmonics.shape[0]

        self.rescale_range = rescale
        self.clips_min = clips_min

        if(isinstance(ranges, tuple)):
            ranges = torch.tensor([ranges[0], ranges[1]], device=device)[None, :].expand(self.func_num, -1)

        self.ranges = ranges.to(self.device) # (B,2)
        self.shifts = ranges[:, 0]  # (B,)
        self.period = ranges[:, 1] - ranges[:, 0]  # (B,)

        # Set the coefficients and harmonics as buffers
        # Change the following to nn.Parameter if we want to train the coefficients
        self.register_buffer(
            "coefficients", coefficients.reshape(coefficients.shape[0], num_harmonics * 2).to(self.device)
        )  # (B,num_harmonics*2)
        self.register_buffer("cos_coeffs", coefficients[:, :, 0].to(self.device))  # (B,num_harmonics)
        self.register_buffer("sin_coeffs", coefficients[:, :, 1].to(self.device))  # (B,num_harmonics)
        self.register_buffer("harmonics", harmonics.to(self.device))  # (B,num_harmonics)

        self.cut_off= cut_off
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given x, evaluates the arbitrary function at the locations determined by x

        Args:
            x : Tensor of shape (N,...). N must match self.func_num
        """
        B, extra_dims = x.shape[0], x.shape[1:]

        assert x.shape[0] == self.func_num or x.shape[0] == 1, (
            "Dim 0 of input must match self.func_num of coefficients or be 1"
        )

        if x.shape[0] == 1:
            x = x.expand(self.func_num, *extra_dims)  # (B,...)


        x = x.reshape(self.func_num, 1, -1)  # (B,1,*)


        func_interior = (
            2 * torch.pi/self.period[:, None, None]
            * self.harmonics[..., None]* (x - self.shifts[:, None, None])
        )
        values = torch.sum(
            self.cos_coeffs[..., None] * torch.cos(func_interior)
            + self.sin_coeffs[..., None] * torch.sin(func_interior),
            dim=1,
        ) # (B,*)

        if self.rescale_range is not None:
            min_vals, _ = values.min(dim=-1, keepdim=True)  # (B,1)
            max_vals, _ = values.max(dim=-1, keepdim=True)  # (B,1)

            values = (values - min_vals) / (max_vals - min_vals + 1e-8)  # (B,*)
            values = values * (self.rescale_range[1] - self.rescale_range[0]) + self.rescale_range[0]
        
        if self.clips_min is not None:
            values[values < self.clips_min] = self.clips_min

        if(self.cut_off):    
            x_out = (x[:,0]<self.ranges[:, 0][:,None]) | (x[:,0]>self.ranges[:, 1][:, None]) # (N,*) true if out of range
            values[x_out] = 0.0 # (B,*) set to 0 if out of range
        
        # Restore initial shape
        values = values.reshape(self.func_num, *extra_dims)

        return values  # (B,*)

    @staticmethod
    def random_arbi(func_num, n_coeffs, ranges, rescale=None, clips_min=None, device="cpu"):
        """
        Returns random arbitrary function, samples coefficients from a Gaussian distribution

        Args:
            func_num : int, number of functions to generate
            n_coeffs : number of coefficients to use for the approximation
            ranges : tuple or (func_num,2) tensor of function x ranges. E.g., if a funciton is defined in [-1,1], the range is (-1,1)
            rescale : tuple, (min,max) range of the x values for the function to be rescaled to
            clips_min : float, after rescaling will clip to this value
            device : torch device
        """
        # Randomly create harmonics and coeffs. Makes sure the coefficients are hermitian s.t. the function is real
        cos_coeffs = torch.randn((func_num, n_coeffs), device=device)
        sin_coeffs = torch.randn((func_num, n_coeffs), device=device)

        harmonics = torch.arange(n_coeffs, device=device).float()  # (num_harmonics,)
        harmonics = harmonics[None].expand(func_num, n_coeffs)  # (func_num,num_harmonics)

        if isinstance(ranges, tuple):
            ranges = torch.tensor([ranges[0], ranges[1]], device=device)[None, :].expand(func_num, -1)

        return ArbitraryFunction(
            coefficients=torch.stack([cos_coeffs, sin_coeffs], dim=-1).reshape(func_num, n_coeffs*2),
            harmonics=harmonics,
            ranges=ranges,
            rescale=rescale,
            clips_min=clips_min,
            device=device,
        )

    @staticmethod
    def from_function_evals(func_evals: torch.Tensor, x_bounds: tuple[float], n_coeffs: int, device="cpu"):
        """
        Given a (batched) tensor of function evaluations, an
        ArbitraryFunction instance approximating the function
        evaluations with n_coeffs harmonics.

        Args:
            func_evals : (N,T), tensor of 1D function evaluations. NOTE:
                This should be the function evaluated at values x_t = t/T*(x_max-x_min) for t=0,1,...,T-1
            x_bounds : tuple, (min,max) range of the x values for the function
            n_coeffs : int, number of harmonics to use for the approximation
        """
        N, T = func_evals.shape

        func_evals = func_evals.to(device)
        ranges = torch.tensor([x_bounds[0], x_bounds[1]], device=device)[None, :].expand(N, -1)  # (N,2)

        fft_coeffs = (
            torch.fft.rfft(func_evals, dim=-1) / T
        )  # We already rescale the coeffs so they work with the inverse
        freqs = torch.fft.rfftfreq(
            T, d=1 / T
        )  # The harmonic frequencies, un-normalized (we scale by period later)

        assert n_coeffs <= freqs.shape[0], (
            f"The number of coefficients {n_coeffs} is too high for the number of samples {freqs.shape[0]}"
        )
        main_freqs = freqs[:n_coeffs]  # (n_coeffs,)The main frequencies we want to keep

        main_freqs = main_freqs[None].expand(N, -1)  # (N,n_coeffs)

        mains_coeffs = fft_coeffs[:, :n_coeffs]  # (N,n_coeffs)
        cos_coeffs = 2 * mains_coeffs.real
        sin_coeffs = -2 * mains_coeffs.imag
        cos_coeffs[:, 0] = mains_coeffs[:, 0].real  # Only exception, no *2 here
        return ArbitraryFunction(
            coefficients=torch.stack([cos_coeffs, sin_coeffs], dim=-1).reshape(N, n_coeffs * 2),
            harmonics=main_freqs,
            ranges=ranges,
            rescale=None,
            clips_min=None,
            device=device,
        )

    @staticmethod
    def from_function(
        function: callable, x_bounds: tuple[float], n_coeffs: int, n_points: int = 1000, device="cpu"
    ):
        """
        Given a callable function that accepts a tensor of shape (T,) of evaluations, computes
        the arbitrary function that approximates the function with 2*n_coeffs+1 harmonics.
        """
        x = (
            torch.arange(n_points, device=device).float() / n_points * (x_bounds[1] - x_bounds[0])
            + x_bounds[0]
        )  # (T,)
        func_evals = function(x)  # (T)

        return ArbitraryFunction.from_function_evals(func_evals[None], x_bounds, n_coeffs, device)
