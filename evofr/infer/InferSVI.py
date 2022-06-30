from typing import Optional, Type
from evofr.data.data_spec import DataSpec
from evofr.models.model_spec import ModelSpec
from evofr.posterior.posterior_handler import PosteriorHandler
from .SVI_handler import SVIHandler
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal


class InferSVI:
    def __init__(
        self,
        iters: int,
        lr: float,
        num_samples: int,
        guide_fn: Type[AutoGuide],
    ):
        self.iters = iters
        self.num_samples = num_samples
        self.handler = SVIHandler(optimizer=Adam(lr))
        self.guide_fn = guide_fn

    def fit(
        self,
        model: ModelSpec,
        data: DataSpec,
        name: Optional[str] = None,
    ) -> PosteriorHandler:
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
        if name is None:
            name = ""
        self.posterior = PosteriorHandler(samples=samples, data=data, name=name)
        return self.posterior


class InferMAP(InferSVI):
    def __init__(self, iters: int, lr: float):
        super().__init__(iters, lr, 1, AutoDelta)


class InferFullRank(InferSVI):
    def __init__(self, iters: int, lr: float, num_samples: int):
        super().__init__(iters, lr, num_samples, AutoMultivariateNormal)
