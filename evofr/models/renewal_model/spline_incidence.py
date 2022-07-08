import jax.numpy as jnp

from evofr.models.model_spec import ModelSpec

from .model_factories import _spline_incidence_model_factory
from .basis_functions import Spline, SplineDeriv


class SplineIncidenceModel(ModelSpec):
    def __init__(
        self,
        k=None,
        CLik=None,
        SLik=None,
    ):
        self.k = k if k is not None else 20
        self.CLik = CLik
        self.SLik = SLik
        self.make_model()

    def make_model(self):
        self.model_fn = _spline_incidence_model_factory(
            self.CLik,
            self.SLik,
        )

    def augment_data(self, data, order=4):
        T = len(data["cases"])
        s = jnp.linspace(0, T, self.k)
        data["X"] = Spline.matrix(jnp.arange(T), s, order=order)
        data["X_prime"] = SplineDeriv.matrix(jnp.arange(T), s, order=order)
