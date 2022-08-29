from jax import random
import jax.numpy as jnp

from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.infer.mcmc import MCMCKernel
from typing import Callable, Dict, Optional, Type


class MCMCHandler:
    def __init__(
        self,
        rng_key=1,
        kernel: Optional[Type[MCMCKernel]] = None,
        **kernel_kwargs
    ):
        """
        Construct MCMC handler.

        Parameters
        ----------
        rng_key:
            seed for pseudorandom number generator.

        kernel:
            optional MCMC kernel. Defaults to NUTS implementation in numpyro.

        **kernel_kwargs:
            kwargs to be passed to MCMC kernel at construction.

        Returns
        ------
        MCMCHandler
        """
        self.rng_key = random.PRNGKey(rng_key)
        self.kernel = kernel if kernel else NUTS
        self.kernel_kwargs = kernel_kwargs
        self.mcmc = None

    def fit(
        self,
        model: Callable,
        data: Dict,
        num_warmup: int,
        num_samples: int,
        **mcmc_kwargs
    ):
        """
        Fit model using MCMC given data.

        model:
            a numpyro model.

        data:
            dictionary containing arguments to 'model'.

        num_warmup:
            number of samples for warmup period in MCMC.

        num_samples:
            number of samples to be returned in MCMC.

        **mcmc_kwargs:
            additional arguments to be passed to MCMC algorithms.
        """
        self.mcmc = MCMC(
            self.kernel(model, **self.kernel_kwargs),
            num_warmup=num_warmup,
            num_samples=num_samples,
            **mcmc_kwargs
        )
        self.mcmc.run(self.rng_key, **data)
        self.samples = self.mcmc.get_samples()

    @property
    def params(self) -> Dict:
        if self.samples is not None:
            return self.samples
        return dict()

    def save_state(self, file_path):
        if self.samples is None:
            return None
        with open(file_path, "wb") as f:
            jnp.save(f, self.samples)

    def load_state(self, file_path):
        with open(file_path, "rb") as f:
            self.samples = jnp.load(f)

    def predict(self, model: Callable, data: Dict, **kwargs):
        predictive = Predictive(model, self.params)
        self.rng_key, rng_key_ = random.split(self.rng_key)
        samples_pred = predictive(rng_key_, pred=True, **data)
        return {**self.params, **samples_pred}
