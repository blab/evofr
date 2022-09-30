from typing import Optional
from jax._src.device_array import DeviceArray
import numpy as np
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist

from evofr.models.renewal_model.basis_functions.basis_fns import BasisFunction
from evofr.models.renewal_model.basis_functions.splines import (
    Spline,
    SplineDeriv,
)

from .model_spec import ModelSpec


def MLR_spline_numpyro(
    seq_counts, N, X, X_deriv, tau=None, pred=False, var_names=None
):
    _, N_variants = seq_counts.shape
    _, N_features = X.shape

    # Sampling parameters
    raw_beta = numpyro.sample(
        "raw_beta",
        dist.Normal(0.0, 3.0),
        sample_shape=(N_features, N_variants - 1),
    )

    beta = numpyro.deterministic(
        "beta",
        jnp.column_stack(
            (raw_beta, jnp.zeros(N_features))
        ),  # All parameters are relative to last column / variant
    )

    logits = jnp.dot(X, beta)  # Logit frequencies by variant

    # Evaluate likelihood
    obs = None if pred else np.nan_to_num(seq_counts)
    numpyro.sample(
        "seq_counts",
        dist.MultinomialLogits(logits=logits, total_count=np.nan_to_num(N)),
        obs=obs,
    )

    # Compute frequency
    numpyro.deterministic("freq", softmax(logits, axis=-1))

    # Compute growth advantage from model
    if tau is not None:
        delta = jnp.dot(X_deriv, beta)
        numpyro.deterministic(
            "ga", jnp.exp(delta[:, :-1] * tau)
        )  # Last row corresponds to linear predictor / growth advantage


class MLRSpline(ModelSpec):
    def __init__(
        self,
        tau: float,
        s: Optional[DeviceArray] = None,
        k: Optional[int] = None,
        order: Optional[int] = None,
    ) -> None:
        """Construct ModelSpec for MLR with a spline basis.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.
        s:
            DeviceArray of knot locations for spline.
            Mutually exclusive argument with k.
        k:
            Number of basis elements.
            Mutually exclusive argument with s.
        order:
            Order of the spline to be used. Defaults to 4.
        Returns
        -------
        MLRSpline
        """
        self.tau = tau  # Fixed generation time
        self.s = s
        self.k = 10 if k is None or s is not None else k
        self.order = 4 if order is None else order
        self.basis_fn = Spline(s=self.s, order=self.order, k=self.k)
        self.basis_fn_deriv = SplineDeriv(s=self.s, order=self.order, k=self.k)
        self.model_fn = MLR_spline_numpyro

    def augment_data(self, data: dict) -> None:
        T = len(data["N"])
        data["X"] = self.basis_fn.make_features(data)
        data["X_deriv"] = self.basis_fn_deriv.make_features(data)
        data["tau"] = self.tau
