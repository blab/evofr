from typing import Optional, Type

from evofr.data.data_spec import DataSpec
from evofr.models.model_spec import ModelSpec
from .MCMC_handler import MCMCHandler
from evofr.posterior.posterior_handler import PosteriorHandler
from numpyro.infer import NUTS
from numpyro.infer.mcmc import MCMCKernel

class InferMCMC:
    def __init__(
        self, num_warmup: int, num_samples: int, kernel: Type[MCMCKernel]
    ):
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.handler = MCMCHandler(kernel=kernel)

    def fit(
        self, model: ModelSpec, data: DataSpec, name: Optional[str] = None
    ) -> PosteriorHandler:
        # Create and augment data dictionary
        input = data.make_data_dict()
        model.augment_data(input)

        # Fit model and retrieve samples
        self.handler.fit(
            model.model_fn, input, self.num_warmup, self.num_samples
        )
        samples = self.handler.predict(model.model_fn, input)

        # Create object to hold posterior samples and data
        if name is None:
            name = ""
        self.posterior = PosteriorHandler(samples=samples, data=data, name=name)
        return self.posterior


class InferNUTS(InferMCMC):
    def __init__(self, num_warmup: int, num_samples: int):
        super().__init__(num_warmup, num_samples, NUTS)
