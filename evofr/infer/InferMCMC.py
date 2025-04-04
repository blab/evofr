from typing import Optional, Type

from numpyro.infer import NUTS
from numpyro.infer.mcmc import MCMCKernel

from evofr.commands.registries import register_inference
from evofr.data.data_spec import DataSpec
from evofr.infer.InferSVI import init_to_MAP
from evofr.models.model_spec import ModelSpec
from evofr.posterior.posterior_handler import PosteriorHandler

from .MCMC_handler import MCMCHandler


class InferMCMC:
    def __init__(
        self,
        num_warmup: int,
        num_samples: int,
        kernel: Type[MCMCKernel],
        **kernel_kwargs
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
        InferMCMC
        """
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.handler = MCMCHandler(kernel=kernel, **kernel_kwargs)

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
        self.handler.fit(model.model_fn, input, self.num_warmup, self.num_samples)
        samples = self.handler.predict(model.model_fn, input)

        # Create object to hold posterior samples and data
        self.posterior = PosteriorHandler(
            samples=samples, data=data, name=name if name is not None else ""
        )
        return self.posterior


@register_inference
class InferNUTS(InferMCMC):
    def __init__(self, num_warmup: int, num_samples: int, **kernel_kwargs):
        super().__init__(num_warmup, num_samples, NUTS, **kernel_kwargs)


@register_inference
class InferNUTS_from_MAP:
    def __init__(self, num_warmup, num_samples, iters, lr):
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.iters = iters
        self.lr = lr

    def fit(self, model, data, name=None):
        init_strat, _ = init_to_MAP(model, data, iters=self.iters, lr=self.lr)
        inference_method = InferNUTS(
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            init_strategy=init_strat,
            dense_mass=True,
        )
        return inference_method.fit(model, data, name=name)
