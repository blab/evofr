from typing import Callable, Optional
from jax import random, lax
from jax import jit
import jax.example_libraries.optimizers as optimizers

from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.svi import SVIState

import pickle


class SVIHandler:
    def __init__(
        self, rng_key=1, loss=Trace_ELBO(num_particles=2), optimizer=None
    ):
        """
        Construct SVI handler.

        Parameters
        ----------
        rng_key:
            seed for pseudorandom number generator.

        loss:
            optional loss to be used for MCMC.

        optimizer:
            optimizer to be used for SVI.

        Returns
        -------
        SVIHandler
        """

        self.rng_key = random.PRNGKey(rng_key)
        self.loss_fn = loss
        self.optimizer = optimizer
        self.svi_state = None

    def init_svi(self, model: Callable, guide: AutoGuide, data: dict):
        self.svi = SVI(model, guide, self.optimizer, self.loss_fn)
        svi_state = self.svi.init(self.rng_key, **data)
        if self.svi_state is None:
            self.svi_state = svi_state
        return self

    def fit(
        self, model: Callable, guide: AutoGuide, data: dict, n_epochs: int
    ):
        self.init_svi(model, guide, data)
        self.svi_result = self.svi.run(
            self.rng_key,
            n_epochs,
            **data,
            progress_bar=False,
            stable_update=True
        )
        self.svi_state = self.svi_result.state

    @property
    def params(self):
        if self.svi and self.svi_state:
            return self.svi.get_params(self.svi_state)

    @property
    def losses(self):
        if self.svi_result:
            return self.svi_result.losses

    def predict(self, model, guide, data, **kwargs):
        if self.svi is None:
            self.init_svi(model, guide, data)

        original_pred = Predictive(guide, params=self.params, **kwargs)
        self.rng_key, rng_key_ = random.split(self.rng_key)
        samples = original_pred(rng_key_, **data)
        predictive = Predictive(model, samples)

        self.rng_key, rng_key_ = random.split(self.rng_key)
        samples_pred = predictive(rng_key_, pred=True, **data)
        return {**samples, **samples_pred}

    def reset_state(self):
        return SVIHandler(self.rng_key, self.loss_fn, self.optimizer)

    # Optim state contains number of iterations and current state
    @property
    def optim_state(self):
        if self.svi_state is not None:
            return self.svi_state.optim_state

    def save_state(self, fp):
        with open(fp, "wb") as f:
            pickle.dump(
                optimizers.unpack_optimizer_state(self.optim_state[1]), f
            )

    def load_state(self, fp):
        with open(fp, "rb") as f:
            optim_state = (0, optimizers.pack_optimizer_state(pickle.load(f)))
        self.svi_state = SVIState(optim_state, None, self.rng_key)
