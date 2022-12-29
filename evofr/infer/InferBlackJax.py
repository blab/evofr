from typing import Callable, Dict, Optional

import jax
from jax import random
from numpyro.infer.util import Predictive, initialize_model

import blackjax

from evofr.data.data_spec import DataSpec
from evofr.models.model_spec import ModelSpec
from evofr.posterior.posterior_handler import PosteriorHandler


# TODO: Make this a shared utility later
def newkey(key):
    key, subkey = random.split(key)
    return subkey


class BlackJaxHandler:
    def __init__(self, kernel, **kernel_kwargs):
        self.kernel_fn = kernel
        self.kernel_kwargs = kernel_kwargs
        self.seed = 100
        self.rng_key = random.PRNGKey(self.seed)
        self.state = None

    @staticmethod
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        keys = random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        return states, infos

    def numpyro_init(self, model, data):
        init_parms, potential_fn_gen, *_ = initialize_model(
            newkey(self.rng_key), model, model_kwargs=data, dynamic_args=True
        )

        def logdenisty_fn(position):
            return -potential_fn_gen(**data)(position)

        initial_position = init_parms.z
        return initial_position, logdenisty_fn

    def run_warmup(
        self, initial_position, logdensity_fn: Callable, num_warmup: int
    ):
        num_warmup = 1 if num_warmup < 1 else num_warmup
        adapt = blackjax.window_adaptation(
            self.kernel_fn,
            logdensity_fn,
            **self.kernel_kwargs,
            num_steps=num_warmup
        )
        last_state, kernel, _ = adapt.run(
            newkey(self.rng_key), initial_position
        )
        return last_state, kernel

    def fit(
        self,
        model: Callable,
        data: Dict,
        num_warmup: int,
        num_samples: int,
    ):
        initial_position, logdensity_fn = self.numpyro_init(model, data)

        # Run adapt window
        if num_warmup > 0:
            starting_state, kernel = self.run_warmup(
                initial_position, logdensity_fn, num_warmup=num_warmup
            )
        else:
            kernel = self.kernel_fn(logdensity_fn, **self.kernel_kwargs)
            starting_state = kernel.init(initial_position)

        # Run sampling
        self.states, self.infos = self.inference_loop(
            newkey(self.rng_key), kernel, starting_state, num_samples
        )

    @property
    def samples(self) -> Dict:
        if self.state is not None:
            return self.state.position
        return dict()

    def predict(self, model, data):
        predictive = Predictive(model, self.states.position)
        samples_pred = predictive(newkey(self.rng_key), pred=True, **data)
        return {**self.states.position, **samples_pred}


class InferBlackJax:
    def __init__(
        self, num_warmup: int, num_samples: int, kernel, **kernel_kwargs
    ):
        """Construct class for specifying MCMC inference method.

        Parameters
        ----------
        num_warmup:
            number of warmup samples to run.

        num_samples:
            number of samples to return from MCMC.

        kernel:
            transition kernel for MCMC.

        Returns
        -------
        InferBlackJax
        """
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.handler = BlackJaxHandler(kernel=kernel, **kernel_kwargs)

    def fit(
        self, model: ModelSpec, data: DataSpec, name: Optional[str] = None
    ) -> PosteriorHandler:
        """Fit model given data using specificed MCMC method.

        Parameters
        ----------
        model:
            ModelSpec for model

        data:
            DataSpec for data to do inference on

        name:
            name used to index posterior

        Returns
        -------
        PosteriorHandler
        """
        # Create and augment data dictionary
        input = data.make_data_dict()
        model.augment_data(input)

        # Fit model and retrieve samples
        self.handler.fit(
            model.model_fn, input, self.num_warmup, self.num_samples
        )
        samples = self.handler.predict(model.model_fn, input)

        # Create object to hold posterior samples and data
        self.posterior = PosteriorHandler(
            samples=samples, data=data, name=name if name is not None else ""
        )
        return self.posterior
