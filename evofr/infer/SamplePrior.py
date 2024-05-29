from typing import Optional

from jax import random

from evofr.data.data_spec import DataSpec
from evofr.infer.backends import Backend
from evofr.infer.InferBlackJax import BlackJaxNumpyro
from evofr.models.model_spec import ModelSpec
from evofr.posterior.posterior_handler import PosteriorHandler


class SamplePrior:
    def __init__(self, num_samples: int, seed: Optional[int] = None):
        self.num_samples = num_samples
        self.seed = seed if seed is not None else 0
        self.rng_key = random.PRNGKey(self.seed)

    def _sample(
        self,
        model: ModelSpec,
        data: DataSpec,
        backend: Optional[Backend] = None,
    ):
        input = data.make_data_dict()
        model.augment_data(input)

        if hasattr(model, "backend") and backend is None:
            backend = model.backend

        # If no backend provided or defined elsewhere
        # default to numpyro
        if backend is None:
            return BlackJaxNumpyro.sample_prior(
                self.rng_key, model, input, self.num_samples
            )

        # Otherwise use provided backend
        if backend == Backend.NUMPYRO:
            return BlackJaxNumpyro.sample_prior(
                self.rng_key, model, input, self.num_samples
            )

    def sample(
        self, model: ModelSpec, data: DataSpec, name: Optional[str] = None
    ) -> PosteriorHandler:
        """
        Sample from model prior.
        """

        samples = self._sample(model, data)

        self.posterior = PosteriorHandler(
            samples=samples, data=data, name=name if name is not None else ""
        )
        return self.posterior
