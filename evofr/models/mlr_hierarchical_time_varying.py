from functools import partial
from typing import Optional
from jax._src.interpreters.batching import Array
import numpy as np
from jax import vmap
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam

from .model_spec import ModelSpec
from evofr.models.renewal_model.basis_functions.splines import (
    Spline,
    SplineDeriv,
)


def hier_MLR_time_numpyro(
    seq_counts,
    N,
    X,
    X_deriv,
    tau=None,
    pool_scale=None,
    pred=False,
    var_names=None,
):
    _, N_variants, N_groups = seq_counts.shape
    _, N_features, _ = X.shape

    pool_scale = 0.1 if pool_scale is None else pool_scale

    # Sampling intercept and fitness parameters
    reparam_config = {
        "beta_loc_step": TransformReparam(),
        "beta_scale": TransformReparam(),
        "alpha": TransformReparam(),
        "raw_beta": TransformReparam(),
    }
    with numpyro.handlers.reparam(config=reparam_config):
        beta_scale = numpyro.sample(
            "beta_scale",
            dist.TransformedDistribution(
                dist.HalfNormal(1.0),
                dist.transforms.AffineTransform(0.0, pool_scale),
            ),
        )
        with numpyro.plate("feature", N_features, dim=-3):
            with numpyro.plate("variants", N_variants - 1, dim=-2):
                # Define loc and scale for fitness beta
                beta_loc_step = numpyro.sample(
                    "beta_loc_step",
                    dist.TransformedDistribution(
                        dist.Normal(0.0, 1.0),
                        dist.transforms.AffineTransform(0.0, 0.2),
                    ),
                )

                beta_loc = numpyro.deterministic(
                    "beta_loc", jnp.cumsum(beta_loc_step, axis=0)
                )  # Turn into random walk

                # Use location and scale parameter to draw within group
                with numpyro.plate("group", N_groups, dim=-1):
                    # Leave intercept alpha unpooled
                    raw_beta = numpyro.sample(
                        "raw_beta",
                        dist.TransformedDistribution(
                            dist.Normal(0.0, 1.0),
                            dist.transforms.AffineTransform(
                                beta_loc, beta_scale
                            ),
                        ),
                    )

    # All parameters are relative to last column / variant
    beta = numpyro.deterministic(
        "beta",
        jnp.concatenate(
            (
                raw_beta,
                jnp.zeros((N_features, 1, N_groups)),
            ),
            axis=1,
        ),
    )

    # (T, F, G) times (F, V, G) -> (T, V, G)
    dot_by_group = vmap(jnp.dot, in_axes=(-1, -1), out_axes=-1)
    logits = dot_by_group(X, beta)  # Logit frequencies by variant

    # Evaluate likelihood
    obs = None if pred else np.swapaxes(np.nan_to_num(seq_counts), 1, 2)

    _seq_counts = numpyro.sample(
        "_seq_counts",
        dist.MultinomialLogits(
            logits=jnp.swapaxes(logits, 1, 2), total_count=np.nan_to_num(N)
        ),
        obs=obs,
    )

    # Re-ordering so groups are last
    seq_counts = numpyro.deterministic(
        "seq_counts", jnp.swapaxes(_seq_counts, 2, 1)
    )

    # Compute frequency
    numpyro.deterministic("freq", softmax(logits, axis=1))

    # Compute growth advantage from model
    delta = numpyro.deterministic(
        "delta", jnp.einsum("tfg, fvg -> tvg", X_deriv, beta[:, :-1, :])
    )
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(delta * tau)
        )  # Last row corresponds to linear predictor / growth advantage
        delta_loc = jnp.einsum(
            "tf, fv -> tv", X_deriv[:, :, 0], beta_loc[:, :, 0]
        )
        numpyro.deterministic("ga_loc", jnp.exp(delta_loc * tau))

    return None


class HierMLRTime(ModelSpec):
    def __init__(
        self,
        tau: float,
        s: Optional[Array] = None,
        k: Optional[int] = None,
        order: Optional[int] = None,
        pool_scale: Optional[float] = None,
    ) -> None:
        """Construct ModelSpec for Hierarchial multinomial logistic regression.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        s:
            Array of knot locations for spline.
            Mutually exclusive argument with k.
        k:
            Number of basis elements.
            Mutually exclusive argument with s.
        order:
            Order of the spline to be used. Defaults to 4.

        pool_scale:
            Prior standard deviation for pooling of growth advantages.

        Returns
        -------
        HierMLR
        """
        self.tau = tau  # Fixed generation time
        self.pool_scale = pool_scale  # Prior std for coefficients

        # Parameters for spline growth advantage
        self.s = s
        self.k = 10 if k is None or s is not None else k
        self.order = 4 if order is None else order
        self.basis_fn = Spline(s=self.s, order=self.order, k=self.k)
        self.basis_fn_deriv = SplineDeriv(s=self.s, order=self.order, k=self.k)

        self.model_fn = partial(
            hier_MLR_time_numpyro, pool_scale=self.pool_scale
        )

    @staticmethod
    def expand_features(X, n_groups):
        """
        Expand design matrix to n_groups.

        Parameters
        ----------
        n_groups:
            number of groups in the hierarchical model.
        """
        return np.stack([X] * n_groups, axis=-1)

    def augment_data(self, data: dict) -> None:
        T, G = data["N"].shape
        data["tau"] = self.tau
        data["X"] = self.expand_features(self.basis_fn.make_features(data), G)
        data["X_deriv"] = self.expand_features(
            self.basis_fn_deriv.make_features(data), G
        )

    def forecast_frequencies(self, samples, forecast_L, tau=None, linear=False, mean_len=7):
        """
        Use posterior beta to forecast posterior frequencies.
        """

        # Let's project based on last values of delta
        # TODO: Add multple options for forecasting based on delta estimates
        # Options to consider: n_avearage, exponential smoothing, .etc
        delta = jnp.array(samples["delta"])
        n_samples, _, _, n_group = delta.shape
        delta_pred = jnp.mean(
            delta[:, -mean_len:, :, :], axis=1
        )  # Using two-week average relative_fitness
        delta_pred = jnp.concatenate(
            (delta_pred, jnp.zeros((n_samples, 1, n_group))), axis=1
        )

        # Extrapolate based on linear approximate delta_pred
        if linear:
            forecast_times = jnp.arange(0., forecast_L) + 1
        else: # Keep delta_pred constant
            forecast_times = jnp.ones(forecast_L)
        delta_pred = (
            delta_pred[:, None, :, :] * forecast_times[None, :, None, None]
        )

        # Creating frequencies from posterior beta
        logits = jnp.log(samples["freq"])[:, -1, :, :]
        logits = (
            logits[:, None, :, :] + jnp.cumsum(delta_pred, axis=1)
        )  # Project based on delta_pred: adding cummulatively in time
        samples["freq_forecast"] = softmax(logits, axis=-2)  # (S, T, V, G)
        if tau is None:
            tau = self.tau
        samples["ga_forecast"] = jnp.exp(delta_pred * tau)
        return samples
