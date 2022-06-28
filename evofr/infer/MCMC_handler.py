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
        if kernel is None:
            kernel = NUTS
        self.rng_key = random.PRNGKey(rng_key)
        self.kernel = kernel
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
        self.mcmc = MCMC(
            self.kernel(model, **self.kernel_kwargs),
            num_warmup=num_warmup,
            num_samples=num_samples,
            **mcmc_kwargs
        )
        self.mcmc.run(self.rng_key, **data)
        self.samples = self.mcmc.get_samples()

    @property
    def params(self):
        if self.samples is not None:
            return self.samples
        return None

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
