from typing import Optional
from evofr.data.data_spec import DataSpec
from evofr.models.model_spec import ModelSpec
from .MCMC_handler import MCMCHandler
from evofr.posterior.posterior_handler import PosteriorHandler
from numpyro.infer import NUTS


class InferMCMC:
    def __init__(self, num_warmup: int, num_samples: int, kernel):
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
        dataset = self.handler.predict(model.model_fn, input)

        # Create object to hold posterior samples and data
        if name is None:
            name = ""
        self.posterior = PosteriorHandler(dataset=dataset, data=data, name="")
        return self.posterior


class InferNUTS(InferMCMC):
    def __init__(self, num_warmup: int, num_samples: int):
        super().__init__(num_warmup, num_samples, NUTS)
