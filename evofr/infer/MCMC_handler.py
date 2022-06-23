from jax import random
import jax.numpy as jnp

from numpyro.infer import NUTS, MCMC, Predictive


class MCMCHandler:
    def __init__(self, rng_key=1, kernel=None):
        if kernel is None:
            kernel = NUTS
        self.rng_key = random.PRNGKey(rng_key)
        self.kernel = kernel
        self.mcmc = None

    def fit(self, model, data, num_warmup, num_samples, **kwargs):
        self.mcmc = MCMC(
            self.kernel(model),
            num_warmup=num_warmup,
            num_samples=num_samples,
            **kwargs
        )
        self.mcmc.run(self.rng_key, **data)
        self.samples = self.mcmc.get_samples()

    @property
    def params(self):
        if self.samples is not None:
            return self.samples
        return None

    def save_state(self, fp):
        if self.samples is None:
            return None
        with open(fp, "wb") as f:
            jnp.save(f, self.samples)

    def load_state(self, fp):
        with open(fp, "rb") as f:
            self.samples = jnp.load(f)

    def predict(self, model, data, **kwargs):
        predictive = Predictive(model, self.params)
        self.rng_key, rng_key_ = random.split(self.rng_key)
        samples_pred = predictive(rng_key_, pred=True, **data)
        return {**self.params, **samples_pred}
