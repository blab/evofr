from functools import partial
from typing import List, Optional

import jax.numpy as jnp

from evofr.models.model_spec import ModelSpec

from .basis_functions import BasisFunction, Spline
from .model_factories import _renewal_model


class RenewalModel(ModelSpec):
    def __init__(
        self,
        g,
        delays,
        seed_L: Optional[int] = None,
        forecast_L: Optional[int] = None,
        k: Optional[int] = None,
        RLik=None,
        CLik=None,
        SLik=None,
        v_names: Optional[List[str]] = None,
        basis_fn: Optional[BasisFunction] = None,
    ):
        self.g_rev = jnp.flip(g, axis=-1)
        self.delays = delays
        self.seed_L = seed_L if seed_L is not None else 7
        self.forecast_L = forecast_L if forecast_L is not None else 0
        self.gen_v_names = v_names

        # Making basis expansion for Rt
        self.k = k if k else 10
        self.basis_fn = basis_fn if basis_fn else Spline(s=None, order=4, k=self.k)

        # Defining model likelihoods
        self.RLik = RLik
        self.CLik = CLik
        self.SLik = SLik
        self.make_model()

    def make_model(self):
        self.model_fn = partial(
            _renewal_model,
            g_rev=self.g_rev,
            delays=self.delays,
            seed_L=self.seed_L,
            forecast_L=self.forecast_L,
            RLik=self.RLik,
            CaseLik=self.CLik,
            SeqLik=self.SLik,
            gen_v_names=self.gen_v_names,
        )

    def augment_data(self, data):
        # Add feature matrix for parameterization of R
        data["X"] = self.basis_fn.make_features(data)
