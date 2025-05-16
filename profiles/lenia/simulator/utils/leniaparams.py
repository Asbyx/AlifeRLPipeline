import torch
import math, os
from .hash_params import params_to_words
from .funcgen import ArbitraryFunction


class BatchParams:
    """
    Class handling general batched parameters.
    It has a dictionary of parameters which are torch tensors (or int) of various
    sizes, with a batch dimension. Specific keys are model specific.
    """

    def __init__(self, param_dict=None, from_file=None, batch_size=None, device="cpu"):
        """
        Args:
            from_file : str, path to file containing parameters. Priority over param_dict
            param_dict : dict, dictionary of parameters
            k_size : int, size of the kernel. Used if both param_dict and from_file are None
            batch_size : int, number of parameters in the batch. Used if both param_dict and from_file are None
            channels : int, number of channels in the automaton
            device : str, device to use
        """
        self.device = device
        self.batch_size = batch_size

        if param_dict is None and from_file is None:
            assert self.batch_size is not None, "batch_size must be provided if no parameters are given"
            param_dict = {}
            print("Warning: initialized empty BatchParams")
        elif from_file is not None:
            param_dict = torch.load(from_file, map_location=device, weights_only=True)

        self.param_dict = {}
        for key in param_dict.keys():
            if isinstance(param_dict[key], torch.Tensor):
                if self.batch_size is None:
                    self.batch_size = param_dict[key].shape[0]
                else:
                    assert self.batch_size == param_dict[key].shape[0], (
                        "All tensors in param_dict must have the same batch size"
                    )
            self.__setattr__(key, param_dict[key])
        if self.batch_size is None:
            print("Couldn't infer batch size from parameters, setting to 1")
            self.batch_size = 1

        self.to(device)

    def __setattr__(self, name, value):
        if not name in {"param_dict", "batch_size", "device"}:
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == self.batch_size, (
                    f"Attempted to add element of incorrect batch size: got {value.shape[0]} expected {self.batch_size}"
                )
            self.param_dict[name] = value

        super().__setattr__(name, value)

    @property
    def name(self):
        """
        Returns a string representation of the parameters.
        """
        return params_to_words(self.param_dict)

    def to(self, device):
        """
        Moves the parameters to a device, like pytorch.
        """
        self.device = device
        for key in self.param_dict.keys():
            if isinstance(self.param_dict[key], torch.Tensor):
                self.param_dict[key] = self.param_dict[key].to(device)

    def save_indiv(self, folder, batch_name=False, annotation=None):
        """
        Saves parameter individually.

        Args:
        folder : path to folder where to save params individually
        params : dictionary of parameters
        batch_name : if True, names indiv parameters with batch name + annotation
        annotation : list of same length as batch_size, an annotation of the parameters.
            Only used if batch_name is True
        """
        os.makedirs(folder, exist_ok=True)

        name = params_to_words(self.param_dict)
        batch_size = self.batch_size

        params_list = [self[i] for i in range(batch_size)]

        if annotation is None:
            annotation = [f"{j:02d}" for j in range(len(params_list))]
        else:
            assert len(annotation) == len(params_list), (
                f"Annotation (len={len(annotation)}) must \
            have same length as batch_size (len={len(params_list)})"
            )

        for j in range(len(params_list)):
            if not batch_name:
                indiv_name = params_to_words(params_list[j].param_dict)
                fullname = indiv_name + ".pt"
            else:
                fullname = name + f"_{annotation[j]}" + ".pt"

            torch.save(params_list[j].param_dict, os.path.join(folder, fullname))

    def save(self, folder, name=None):
        """
        Saves the (batched) parameters to a folder.
        """
        os.makedirs(folder, exist_ok=True)
        if name is None:
            name = params_to_words(self.param_dict)

        torch.save(self.param_dict, os.path.join(folder, name + ".pt"))

    def load(self, path):
        """
        Loads the parameters from a file.
        """
        self.__init__(from_file=path)

    def __mul__(self, scalar: float) -> "BatchParams":
        """
        Multiplies all parameters by a scalar.
        """
        if not isinstance(scalar, (int, float)):
            raise ValueError("Can only multiply by a scalar")
        new_params = {}
        for key in self.param_dict.keys():
            if not isinstance(self.param_dict[key], torch.Tensor):
                # Assume non-tensor parameters are not to be multiplied
                new_params[key] = self.param_dict[key]
            else:
                new_params[key] = self.param_dict[key] * scalar

        return type(self)(param_dict=new_params, device=self.device)

    def __rmul__(self, scalar: float) -> "BatchParams":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "BatchParams":
        return self.__mul__(1.0 / scalar)

    def __contains__(self, key):
        return key in self.param_dict.keys()

    def __getitem__(self, idx):
        """
        Works as a dictionary, indexing the parameters with strings, or
        as advanced indexing on the batch dimension, like in pytorch.
        Will ALWAYS keep at least one dimension for the batch size.
        In other words,params[1] is the same as params[1:2]

        WARNING : will not fail if the key is not found, will return None.
        """
        if isinstance(idx, str):
            return self.param_dict.get(idx, None)  # Soft fail if key not found, return None.
        elif isinstance(idx, int):
            idx = slice(idx, idx + 1)

        params = {}
        for k, v in self.param_dict.items():
            if not isinstance(v, torch.Tensor):
                params[k] = v
            else:
                params[k] = v[idx]

        return type(self)(params, device=self.device)

    def __setitem__(self, idx, value):
        """
        Works as a dictionary, setting the parameters with strings, or
        as advanced indexing on the batch dimension, like in pytorch.
        Will ALWAYS keep at least one dimension for the batch size.
        In other words,params[1] is the same as params[1:2]
        """
        if isinstance(idx, str):
            self.__setattr__(idx, value)
        else:
            # if idx is not a string, we assume value is BatchParams
            assert isinstance(value, BatchParams), (
                "Can only setitem with BatchParams when using advanced indexing"
            )
            assert value.param_dict.keys() == self.param_dict.keys(), (
                "Keys of the two BatchParams do not match"
            )
            if isinstance(idx, int):
                idx = slice(idx, idx + 1)

            # Else, assume its some advanced pytorch indexing
            for k, v in value.param_dict.items():
                if not isinstance(v, torch.Tensor):
                    self.__setattr__(k, v)
                else:
                    self.param_dict[k][idx] = v

    def __add__(self, other: "BatchParams") -> "BatchParams":
        """
        Adds two sets of parameters together.
        """
        assert self.batch_size == other.batch_size, "Batch sizes do not match"

        new_params = {}
        for key in self.param_dict.keys():
            if not isinstance(self.param_dict[key], torch.Tensor):
                assert self.param_dict[key] == other.param_dict[key], (
                    f"Non-tensor parameters do not match for key {key}"
                )
                new_params[key] = self.param_dict[key]
            else:
                new_params[key] = self.param_dict[key] + other.param_dict[key]

        return type(self)(param_dict=new_params, device=self.device)

    def expand(self, batch_size) -> "BatchParams":
        """
        Expands parameters with batch_size 1 to a larger batch size.

        Args:
            batch_size : int, new batch size
        """
        assert self.batch_size == 1, f"Batch size must be 1 to expand, here is {self.batch_size}"

        new_params = {}

        for key in self.param_dict.keys():
            update = self.param_dict[key]
            if not isinstance(update, torch.Tensor):
                new_params[key] = update
            else:
                n_d = len(update.shape) - 1
                new_params[key] = update.repeat(batch_size, *([1] * n_d))

        return type(self)(param_dict=new_params, device=self.device)

    def cat(self, other: "BatchParams") -> "BatchParams":
        """
        Concatenates two sets of parameters together.
        """
        assert self.k_size == other.k_size, "Kernel sizes do not match"
        assert self.device == other.device, f"Devices do not match, got {self.device} and {other.device}"
        new_params = {}
        for key in self.param_dict.keys():
            if not isinstance(self.param_dict[key], torch.Tensor):
                assert self.param_dict[key] == other.param_dict[key], (
                    f"Non-tensor parameters do not match for key {key}"
                )
                new_params[key] = self.param_dict[key]
            else:
                new_params[key] = torch.cat([self.param_dict[key], other.param_dict[key]], dim=0)

        return type(self)(param_dict=new_params, device=self.device)

    def mutate(self, magnitude=0.02, rate=0.1, frozen_keys=[]) -> "BatchParams":
        """
        Mutates the parameters by a small amount.

        Args:
            magnitude : float, magnitude of the mutation
            rate : float, will change a parameter with this rate
            frozen_keys : list of str, keys to not mutate
        """
        keys = list(self.param_dict.keys())

        new_params = {}
        for key in keys:
            if key not in frozen_keys and isinstance(self.param_dict[key], torch.Tensor):
                tentative = self.param_dict[key] * (
                    1 + magnitude * torch.randn_like(self.param_dict[key], dtype=torch.float32)
                )

                new_params[key] = torch.where(
                    torch.rand_like(tentative) < rate, tentative, self.param_dict[key]
                )
            else:
                new_params[key] = self.param_dict[key]

        return type(self)(param_dict=new_params, device=self.device)


class LeniaParams(BatchParams):
    """
    Class handling Lenia parameters

    Keys:
        'k_size' : odd int, size of kernel used for computations
        'mu' : (B,C*k_mult,C) tensor, mean of growth functions
        'sigma' : (B,C*k_mult,C) tensor, standard deviation of the growth functions
        'beta' :  (B,C*k_mult,C, # of rings) float, max of the kernel rings
        'mu_k' : (B,C*k_mult,C, # of rings) [0,1.], location of the kernel rings
        'sigma_k' : (B,C*k_mult,C, # of rings) float, standard deviation of the kernel rings
        'weights' : (B,C*k_mult,C) float, weights for the growth weighted sum
        'k_harmonics' : (B,C*k_mult,C, # of harmonics) floats, harmonics frequencies for the kernel (optional)
        'k_coeffs' : (B,C*k_mult,C, # of harmonics) floats, harmonics coefficients for the kernel (optional)
        'k_rescale' : tuple of floats, rescale range for the kernel (optional)
        'k_clip' : float, clip value for the kernel (optional)
        'g_harmonics' : (B,C*k_mult,C, # of harmonics) floats, harmonics frequencies for the growth (optional)
        'g_coeffs' : (B,C*k_mult,C, # of harmonics) floats, harmonics coefficients for the growth (optional)
        'g_rescale' : tuple of floats, rescale range for the growth (optional)
        'g_clip' : float, clip value for the growth (optional)
    """

    def __init__(
        self,
        param_dict=None,
        from_file=None,
        k_size=None,
        batch_size=None,
        channels=3,
        k_mult=1,
        device="cpu",
    ):
        """
        Args:
            from_file : str, path to file containing parameters. Priority over param_dict
            param_dict : dict, dictionary of parameters
            k_size : int, size of the kernel. Used if both param_dict and from_file are None
            batch_size : int, number of parameters in the batch. Used if both param_dict and from_file are None
            channels : int, number of channels in the automaton
            device : str, device to use
        """
        if param_dict is None and from_file is None:
            assert k_size is not None and batch_size is not None, (
                "k_size and batch_size must be provided if no parameters are given"
            )

            param_dict = LeniaParams.default_gen(
                batch_size=batch_size, num_channels=channels, k_size=k_size, k_mult=k_mult, device=device
            ).param_dict  # dis very ugly but not sure how to do it better
            super().__init__(param_dict=param_dict, device=device)
        else:
            super().__init__(param_dict=param_dict, from_file=from_file, batch_size=batch_size, device=device)

        assert "weights" in self.param_dict.keys(), 'LeniaParams need "weights" tensor'
        assert "k_size" in self.param_dict.keys(), 'LeniaParams need "k_size" value'
        if "k_mult" not in self.param_dict.keys():
            self.k_mult = 1  # legacy param that has no k_mult
        if "num_channels" not in self.param_dict.keys():
            self.num_channels = channels  # legacy param that has no num_channels
        self._sanitize()
        self.to(device)

    def _sanitize(self):
        """
        Sanitizes the parameters by clamping them to valid values.
        """
        param_keys = self.param_dict.keys()
        if "mu" in param_keys:
            self.mu = torch.clamp(self.mu, -2, 2)
        if "sigma" in param_keys:
            self.sigma = torch.clamp(self.sigma, 1e-4, 1.0)
        if "beta" in param_keys:
            self.beta = torch.clamp(self.beta, 0, None)
        if "mu_k" in param_keys:
            self.mu_k = torch.clamp(self.mu_k, 0.0, 2.0)
        if "sigma_k" in param_keys:
            self.sigma_k = torch.clamp(self.sigma_k, 1e-4, 1.0)

        self.weights = torch.clamp(self.weights, 0, None)
        ## Normalize weights
        N = self.weights.sum(dim=1, keepdim=True)  # (B,1,C)
        self.weights = torch.where(N > 1.0e-6, self.weights / N, 0)

    def reroll_params(self, kernel=True, arbi=False):
        """
        Rerolls part of the parameters (growth or kernels).

        Args:
            kernels : if True, rerolls the kernels, otherwise rerolls the growth functions
            arbi : if True, rerolls with arbitrary functions, otherwise rerolls with random_gen
        """
        if kernel:
            if arbi:
                from_fourier = LeniaParams._fourier_master(
                    batch_size=self.batch_size,
                    k_mult=self.k_mult,
                    num_channels=self.param_dict["num_channels"],
                    decay=0.3,
                    harmonics=(1.0, 4),
                    ranges=(0.0, 1.0),
                    rescale=(-0.7, 1.0),
                    clip=0.0,
                    device=self.device,
                )
                self.k_coeffs = from_fourier["coeffs"]
                self.k_harmonics = from_fourier["harmonics"]
                self.k_rescale = from_fourier["rescale"]
                self.k_clip = from_fourier["clip"]
            else:
                from_random_gen = LeniaParams.random_gen(
                    batch_size=self.batch_size,
                    num_channels=self.num_channels,
                    k_mult=self.k_mult,
                    k_size=self.k_size,
                    device=self.device,
                ).param_dict
                self.mu_k = from_random_gen["mu_k"]
                self.sigma_k = from_random_gen["sigma_k"]
                self.beta = from_random_gen["beta"]
        else:
            if arbi:
                from_fourier = LeniaParams._fourier_master(
                    batch_size=self.batch_size,
                    num_channels=self.param_dict["num_channels"],
                    k_mult=self.k_mult,
                    decay=0.2,
                    harmonics=(1.0, 3),
                    ranges=(0.0, 2.0),
                    rescale=(-1.0, 1.0),
                    clip=-0.5,
                    device=self.device,
                )
                self.g_coeffs = from_fourier["coeffs"]
                self.g_harmonics = from_fourier["harmonics"]
                self.g_rescale = from_fourier["rescale"]
                self.g_clip = from_fourier["clip"]
            else:
                from_random_gen = LeniaParams.random_gen(
                    batch_size=self.batch_size,
                    num_channels=self.num_channels,
                    k_mult=self.k_mult,
                    k_size=self.k_size,
                    device=self.device,
                ).param_dict
                self.mu = from_random_gen["mu"]
                self.sigma = from_random_gen["sigma"]

        self._sanitize()

    @staticmethod
    def mixed_gen(
        batch_size,
        num_channels=3,
        sigma_size=1.0,
        k_size=None,
        k_mult=1,
        k_arbi=True,
        g_arbi=False,
        k_coeffs=3,
        k_rescale=(-0.3, 1.0),
        g_coeffs=4,
        g_rescale=(-1.0, 1.0),
        g_clip=None,
        device="cpu",
    ):
        """
        Mixed generation between arbitrary and standard Lenia parameters.
        With k_arbi and g_arbi set to False, it is the same as random_gen.
        """
        params = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )

        from_random_gen = LeniaParams.random_gen(
            batch_size=batch_size,
            num_channels=num_channels,
            sigma_size=sigma_size,
            k_size=k_size,
            k_mult=k_mult,
            device=device,
        ).param_dict
        # from_random_gen = LeniaParams.default_gen(batch_size=batch_size,num_channels=num_channels,k_size=k_size,device=device).param_dict
        if k_arbi:
            from_fourier = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                k_mult=k_mult,
                decay=None,
                harmonics=(0.0, k_coeffs),
                ranges=(0.0, 1.0),
                rescale=k_rescale,
                clip=0.0,
                device=device,
            )
            params["k_coeffs"] = from_fourier["coeffs"]
            params["k_harmonics"] = from_fourier["harmonics"]
            params["k_rescale"] = from_fourier["rescale"]
            params["k_clip"] = from_fourier["clip"]
        else:
            params["mu_k"] = from_random_gen["mu_k"]
            params["sigma_k"] = from_random_gen["sigma_k"]
            params["beta"] = from_random_gen["beta"]

        if g_arbi:
            from_fourier = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                decay=None,
                harmonics=(0.0, g_coeffs),
                ranges=(0.0, 2.0),
                k_mult=k_mult,
                clip=g_clip,
                rescale=g_rescale,
                device=device,
            )
            params["g_coeffs"] = from_fourier["coeffs"]
            params["g_harmonics"] = from_fourier["harmonics"]
            params["g_rescale"] = from_fourier["rescale"]
            params["g_clip"] = from_fourier["clip"]
        else:
            params["mu"] = from_random_gen["mu"]
            params["sigma"] = from_random_gen["sigma"]

        return LeniaParams(params, device=device)

    @staticmethod
    def exp_decay_gen(
        batch_size,
        num_channels=3,
        k_arbi=True,
        g_arbi=False,
        k_size=None,
        k_mult=1,
        k_decay=1.0,
        k_harmo_start=1.0,
        g_decay=1.0,
        g_harmo_start=1.0,
        k_coeffs=6,
        k_rescale=(-0.3, 1.0),
        g_coeffs=4,
        g_rescale=(-2.0, 2.0),
        g_clip=None,
        device="cpu",
    ):
        """
        Generates parameters with exponential decay.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """
        params = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )

        from_random_gen = LeniaParams.random_gen(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        ).param_dict

        if k_arbi:
            from_exp_gen = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                decay=k_decay,
                k_mult=k_mult,
                harmonics=(k_harmo_start, k_coeffs),
                ranges=(0.0, 1.0),
                rescale=k_rescale,
                clip=0.0,
                device=device,
            )

            for key, value in from_exp_gen.items():
                params["k_" + key] = value
        else:
            for key in ["mu_k", "sigma_k", "beta"]:
                params[key] = from_random_gen[key]

        if g_arbi:
            from_exp_gen = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                decay=g_decay,
                k_mult=k_mult,
                harmonics=(g_harmo_start, g_coeffs),
                ranges=(0.0, 2.0),
                rescale=g_rescale,
                clip=g_clip,
                device=device,
            )

            for key, value in from_exp_gen.items():
                params["g_" + key] = value
        else:
            params["mu"] = from_random_gen["mu"]
            params["sigma"] = from_random_gen["sigma"]

        return LeniaParams(params, device=device)

    @staticmethod
    def default_gen(batch_size, num_channels=3, k_size=None, k_mult=1, device="cpu"):
        """
        Generates (standard) parameters with random values. Prior tuned to
        be close to 'stability' in Lenia.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """

        params_base = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )

        mu = 0.7 * torch.rand((batch_size, num_channels * k_mult, num_channels), device=device)
        sigma = (
            mu
            / (math.sqrt(2 * math.log(2)))
            * 0.8
            * torch.rand((batch_size, num_channels * k_mult, num_channels), device=device)
            + 1e-4
        )

        params = {
            "mu": mu,
            "sigma": sigma,
            "beta": torch.rand((batch_size, num_channels * k_mult, num_channels, 3), device=device),
            "mu_k": 0.5
            + 0.2 * torch.randn((batch_size, num_channels * k_mult, num_channels, 3), device=device),
            "sigma_k": 0.05
            * (
                1
                + torch.clamp(
                    0.3 * torch.randn((batch_size, num_channels * k_mult, num_channels, 3), device=device),
                    min=-0.9,
                )
                + 1e-4
            ),
        }

        params.update(params_base)
        return LeniaParams(params, device=device)

    @staticmethod
    def random_gen(batch_size, num_channels=3, k_size=None, k_mult=1, sigma_size=1.0, device="cpu"):
        """
        Full random generation for standard Lenia Parameters. Weights are biased towards
        self-interaction.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """
        params_base = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )
        mu = torch.rand((batch_size, num_channels * k_mult, num_channels), device=device)
        sigma = (
            sigma_size * torch.rand((batch_size, num_channels * k_mult, num_channels), device=device) + 1e-4
        )

        params = {
            "mu": mu,
            "sigma": sigma,
            "beta": torch.rand((batch_size, num_channels * k_mult, num_channels, 3), device=device),
            "mu_k": torch.rand((batch_size, num_channels * k_mult, num_channels, 3), device=device),
            "sigma_k": 0.05
            * (
                1
                + torch.clamp(
                    0.3 * torch.randn((batch_size, num_channels * k_mult, num_channels, 3), device=device),
                    min=-0.9,
                )
                + 1e-4
            ),
        }
        params.update(params_base)

        return LeniaParams(params, device=device)

    @staticmethod
    def fourier_range_gen(
        batch_size,
        num_channels=3,
        k_size=None,
        k_mult=1,
        k_arbi=True,
        g_arbi=False,
        k_harmonics: torch.Tensor = torch.tensor([2, 3, 4]),
        k_rescale=(-0.3, 1.0),
        g_harmonics: torch.Tensor = torch.tensor([1, 2]),
        g_rescale=(-1.0, 1.0),
        g_clip=None,
        device="cpu",
    ):
        """
        Generates parameters with Fourier range.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """
        params = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )

        from_random_gen = LeniaParams.random_gen(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        ).param_dict

        if k_arbi:
            from_fourier = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                harmonics=k_harmonics,
                k_mult=k_mult,
                ranges=(0.0, 1.0),
                rescale=k_rescale,
                device=device,
            )
            params["k_coeffs"] = from_fourier["coeffs"]
            params["k_harmonics"] = from_fourier["harmonics"]
            params["k_rescale"] = from_fourier["rescale"]
            params["k_clip"] = from_fourier["clip"]
        else:
            params["mu_k"] = from_random_gen["mu_k"]
            params["sigma_k"] = from_random_gen["sigma_k"]
            params["beta"] = from_random_gen["beta"]

        if g_arbi:
            from_fourier = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                harmonics=g_harmonics,
                k_mult=k_mult,
                ranges=(0.0, 2.0),
                rescale=g_rescale,
                clip=g_clip,
                device=device,
            )
            params["g_coeffs"] = from_fourier["coeffs"]
            params["g_harmonics"] = from_fourier["harmonics"]
            params["g_rescale"] = from_fourier["rescale"]
            params["g_clip"] = from_fourier["clip"]
        else:
            params["mu"] = from_random_gen["mu"]
            params["sigma"] = from_random_gen["sigma"]

        return LeniaParams(params, device=device)

    @staticmethod
    def to_arbi_params(
        lenia_params: "LeniaParams", discretization_points=1000, device="cpu"
    ) -> "LeniaParams":
        """
        Convert standard Lenia parameters to arbitrary function parameters.

        Args:
            lenia_params : LeniaParams, parameters to convert
            discretization_points : int, number of points to use for the discretization of the functions
            device : str, device to use
        """

        def k_func(mu_k, sigma_k, beta):
            x_range = torch.arange(
                0.0, 1, step=1 / discretization_points, device=device
            )  # (discretization_points,)
            x_range = x_range[None, None, None, None, :]  # (1,1,1,1,discretization_points)

            K = torch.exp(
                -(((x_range - mu_k[..., None]) / sigma_k[..., None]) ** 2) / 2.0
            )  # (B,C,C,#of rings, discretization_points)
            beta = beta[..., None]  # (B,C,C,#of rings, 1)
            K = torch.sum(beta * K, dim=-2)  # (B,C,C,discretization_points)

            return K

        def g_func(mu, sigma):
            x_range = torch.arange(
                0, 2, step=2 / discretization_points, device=device
            )  # (discretization_points,)
            x_range = x_range[None, None, None, :]  # (1,1,1,discretization_points)

            G = (
                2 * torch.exp(-(((x_range - mu[..., None]) / sigma[..., None]) ** 2) / 2.0) - 1
            )  # (B,C,C,discretization_points)
            return G

        k_mult = lenia_params.param_dict.get("k_mult", 1)
        if "mu" in lenia_params:
            B, _, C = lenia_params.mu.shape
            g_evals = g_func(lenia_params.mu, lenia_params.sigma)  # (B,C,C,discretization_points)
            g_evals = g_evals.reshape(B * C * k_mult * C, discretization_points)
            g_coeffs = 8
            g_arbi = ArbitraryFunction.from_function_evals(
                g_evals, (0.0, 2.0), n_coeffs=g_coeffs, device=device
            )
            g_coeffs_val = g_arbi.coefficients.reshape(B, C * k_mult, C, 2 * g_coeffs)
            g_harmonics = g_arbi.harmonics.reshape(B, C * k_mult, C, g_coeffs)
        else:
            g_coeffs_val = lenia_params.g_coeffs
            g_harmonics = lenia_params.g_harmonics

        if "mu_k" in lenia_params:
            B, _, C, _ = lenia_params.mu_k.shape
            k_evals = k_func(
                lenia_params.mu_k, lenia_params.sigma_k, lenia_params.beta
            )  # (B,C,C,discretization_points)
            k_evals = k_evals.reshape(B * C * k_mult * C, discretization_points)
            k_coeffs = 15
            k_arbi = ArbitraryFunction.from_function_evals(
                k_evals, (0.0, 1.0), n_coeffs=k_coeffs, device=device
            )
            k_coeffs_val = k_arbi.coefficients.reshape(B, C * k_mult, C, 2 * k_coeffs)
            k_harmonics = k_arbi.harmonics.reshape(B, C * k_mult, C, k_coeffs)
        else:
            k_coeffs_val = lenia_params.k_coeffs
            k_harmonics = lenia_params.k_harmonics

        params = {
            "k_size": lenia_params.k_size,
            "k_mult": k_mult,
            "k_coeffs": k_coeffs_val,
            "k_harmonics": k_harmonics,
            "g_coeffs": g_coeffs_val,
            "g_harmonics": g_harmonics,
            "weights": lenia_params.weights,
        }

        return LeniaParams(params, device=device)

    # ========== BELOW, UTILITY FUNCTIONS FOR GENERATING ARBI PARAMETERS ==========
    @staticmethod
    def _fourier_master(
        batch_size,
        num_channels,
        harmonics,
        ranges,
        k_mult=1,
        decay=None,
        rescale=None,
        clip=None,
        device="cpu",
    ):
        """
        Master fourier function generator. Has all the options for generating Fourier functions parameters.

        Args:
            batch_size : number of parameters to generate
            num_channels : number of channels in the automaton
            harmonics : determines the harmonics to be used. Has many modes listed below:
                - tuple (harmo_start, num_harmonics) : will generate num_harmonics harmonics starting from harmo_start
                - float tensor (num_harmonics,) : will use the harmonics in the specified tensor, for all batches and channels
                - float tensor (batch_size*num_channels*num_channels,num_harmonics) : will use the harmonics in the specified tensor, can be different for each batch and channel
            (num_harmonics,) integers or (batch_size*num_channels*num_channels,num_harmonics) harmonics to be used
            decay : float, decay rate for the exponential function
            ranges : tuple, x-range for the functions. NOTE : could support different ranges for the functions, but not useful right now.
            rescale : tuple, (min,max) range of the x values for the function to be rescaled to
            clip : float, after rescaling will clip to this value
            device : device on which to generate the parameters
        """
        if isinstance(harmonics, tuple):
            harmo_start, num_harmonics = harmonics
            harmonics = torch.arange(harmo_start, harmo_start + num_harmonics, device=device)[None, :].expand(
                batch_size * num_channels * k_mult * num_channels, num_harmonics
            )
        elif isinstance(harmonics, torch.Tensor):
            if len(harmonics.shape) == 1:
                harmonics = harmonics[None, :].expand(
                    batch_size * num_channels * k_mult * num_channels, num_harmonics
                )
            elif len(harmonics.shape) == 2:
                assert harmonics.shape[0] == batch_size * num_channels * num_channels * k_mult, (
                    f"Expected harmonics tensor of shape ({batch_size * num_channels * num_channels * k_mult},{harmonics.shape[1]}), got {harmonics.shape}"
                )

        num_harmonics = harmonics.shape[1]
        coeffs = torch.randn(
            (batch_size, num_channels * k_mult, num_channels, num_harmonics, 2), device=device
        )  # (batch_size,num_channels,num_channels,num_harmonics,2)
        if decay is not None:
            dampening = torch.exp(-decay * torch.arange(0, num_harmonics, device=device).float())[
                None, None, None, :, None
            ]
            coeffs = coeffs * dampening

        coeffs = coeffs.reshape(
            (batch_size, num_channels * k_mult, num_channels, num_harmonics * 2)
        )  # (batch_size,num_channels,num_channels,num_harmonics*2)
        harmonics = harmonics.reshape(
            batch_size, num_channels * k_mult, num_channels, num_harmonics
        )  # (batch_size,num_channels,num_channels,num_harmonics)
        return {
            "coeffs": coeffs,
            "harmonics": harmonics,
            "rescale": rescale,
            "clip": clip,
            "ranges": ranges,
        }

    @staticmethod
    def _universal_params(batch_size, num_channels, k_size=31, k_mult=1, diag_inhibition=0.8, device="cpu"):
        """
        Return bare minimum k_size and weights dictionary for the model.

        Args:
            batch_size : number of parameters to generate
            num_channels : number of channels in the automaton
            k_size : int, size of the kernel
            k_mult : int, multiplier for the number of source channels
            diag_inhibition : float, value of the diagonal inhibition
            device : device on which to generate the parameters
        """
        if k_mult == 1:
            return {
                "k_size": k_size,
                "weights": torch.rand(batch_size, num_channels, num_channels, device=device)
                * (1 - diag_inhibition * torch.diag(torch.ones(num_channels, device=device))),
                "k_mult": k_mult,
                "num_channels": num_channels,
            }
        else:
            # No diag inhibition possible for k_mult > 1
            return {
                "k_size": k_size,
                "weights": torch.rand(batch_size, num_channels * k_mult, num_channels, device=device),
                "k_mult": k_mult,
                "num_channels": num_channels,
            }


if __name__ == "__main__":
    test = LeniaParams(batch_size=1, k_size=21)
