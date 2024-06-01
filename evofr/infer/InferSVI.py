from typing import Optional, Type

import jax.numpy as jnp
from numpyro.infer import init_to_value
from numpyro.infer.autoguide import AutoDelta, AutoGuide, AutoMultivariateNormal
from numpyro.optim import Adam

from evofr.data.data_spec import DataSpec
from evofr.models.model_spec import ModelSpec
from evofr.posterior.posterior_handler import PosteriorHandler

from .SVI_handler import SVIHandler


class InferSVI:
    def __init__(
        self,
        iters: int,
        lr: float,
        num_samples: int,
        guide_fn: Type[AutoGuide],
        **handler_kwargs,
    ):
        """Construct class for specifying SVI inference method.

        Parameters
        ----------
        iters:
            number of iterations to run optimizer.

        lr:
            learning rate for optimizer

        num_samples:
            number of samples to return from approximate posterior.

        guide_fn:
            variational model or guide to follow for SVI.

        handler_kwargs:
            keyword arguments that can be passed to SVIHandler such as
            `loss_fn` (instance of TraceElbo) or `optimizer`.

        Returns
        -------
        InferSVI
        """
        self.iters = iters
        self.num_samples = num_samples
        self.handler = SVIHandler(optimizer=Adam(lr), **handler_kwargs)
        self.guide_fn = guide_fn

    def fit(
        self,
        model: ModelSpec,
        data: DataSpec,
        name: Optional[str] = None,
    ) -> PosteriorHandler:
        """Fit model given data using specificed SVI method.

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

        # Create guide for SVI
        guide = self.guide_fn(model.model_fn)

        # Fit model and retrieve samples
        self.handler.fit(model.model_fn, guide, input, self.iters)
        samples = self.handler.predict(
            model.model_fn, guide, input, num_samples=self.num_samples
        )
        samples["losses"] = self.handler.losses

        # Create object to hold posterior samples and data
        self.posterior = PosteriorHandler(
            samples=samples, data=data, name=name if name is not None else ""
        )
        return self.posterior


class InferMAP(InferSVI):
    def __init__(self, iters: int, lr: float, **handler_kwargs):
        super().__init__(iters, lr, 1, AutoDelta, **handler_kwargs)


class InferFullRank(InferSVI):
    def __init__(self, iters: int, lr: float, num_samples: int, **handler_kwargs):
        super().__init__(
            iters, lr, num_samples, AutoMultivariateNormal, **handler_kwargs
        )


def init_to_MAP(
    model: ModelSpec,
    data: DataSpec,
    iters: int = 10_000,
    lr: float = 4e-3,
):
    """
    Initilization strategy for MCMC.
    Estimates MAP for given model and data.
    Returns initilization strategy and MAP estimates.
    """

    # Fit MAP with given model and data
    infer_map = InferMAP(iters=iters, lr=lr)
    MAP = infer_map.fit(model, data)

    # Return initilization strategy for MCMC and MAP estimates
    samples = {
        k: jnp.squeeze(v, axis=0) if v.shape[0] == 1 else v
        for k, v in MAP.samples.items()
    }
    return init_to_value(values=samples), MAP
