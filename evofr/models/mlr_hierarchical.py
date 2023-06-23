from functools import partial
from typing import Optional
import numpy as np
from jax import vmap
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam

from .model_spec import ModelSpec
from .multinomial_logistic_regression import (
    MultinomialLogisticRegression,
    simulate_MLR,
)


def simulate_hier_mlr(growth_advantages, freq0, tau, Ns):
    # Assume all are in list of size groups
    assert len(growth_advantages) == len(freq0)
    assert len(growth_advantages) == len(Ns)
    groups = len(growth_advantages)
    seq_counts = []
    for group in range(groups):
        _, sc_group = simulate_MLR(
            growth_advantages[group], freq0[group], tau, Ns[group]
        )
        seq_counts.append(sc_group)
    return seq_counts


def hier_MLR_numpyro(
    seq_counts, N, X, tau=None, pool_scale=None, pred=False, var_names=None
):
    _, N_variants, N_groups = seq_counts.shape
    _, N_features, _ = X.shape

    pool_scale = 0.1 if pool_scale is None else pool_scale

    # Sampling intercept and fitness parameters
    with numpyro.plate("variants", N_variants - 1, dim=-2):

        # Define loc and scale for fitness beta
        reparam_config = {
            "beta_loc": TransformReparam(),
            "beta_scale": TransformReparam(),
            "alpha": TransformReparam(),
            "raw_beta": TransformReparam(),
        }
        with numpyro.handlers.reparam(config=reparam_config):
            beta_loc = numpyro.sample(
                "beta_loc",
                dist.TransformedDistribution(
                    dist.Normal(0.0, 1.0),
                    dist.transforms.AffineTransform(0.0, 0.2),
                ),
            )
            beta_scale = numpyro.sample(
                "beta_scale",
                dist.TransformedDistribution(
                    dist.HalfNormal(1.0),
                    dist.transforms.AffineTransform(0.0, pool_scale),
                ),
            )

            # Use location and scale parameter to draw within group
            with numpyro.plate("group", N_groups, dim=-1):
                # Leave intercept alpha unpooled
                raw_alpha = numpyro.sample(
                    "alpha",
                    dist.TransformedDistribution(
                        dist.Normal(0.0, 1.0),
                        dist.transforms.AffineTransform(0.0, 6.0),
                    ),
                )
                raw_beta = numpyro.sample(
                    "raw_beta",
                    dist.TransformedDistribution(
                        dist.Normal(0.0, 1.0),
                        dist.transforms.AffineTransform(beta_loc, beta_scale),
                    ),
                )

    # All parameters are relative to last column / variant
    beta = numpyro.deterministic(
        "beta",
        jnp.concatenate(
            (
                jnp.stack((raw_alpha, raw_beta)),
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
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(beta[-1, :-1, :] * tau)
        )  # Last row corresponds to linear predictor / growth advantage
        numpyro.deterministic("ga_loc", jnp.exp(beta_loc[:, 0] * tau))


class HierMLR(ModelSpec):
    def __init__(self, tau: float, pool_scale: Optional[float] = None) -> None:
        """Construct ModelSpec for Hierarchial multinomial logistic regression.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        pool_scale:
            Prior standard deviation for pooling of growth advantages.

        Returns
        -------
        HierMLR
        """
        self.tau = tau  # Fixed generation time
        self.pool_scale = pool_scale  # Prior std for coefficients
        self.model_fn = partial(hier_MLR_numpyro, pool_scale=self.pool_scale)

    @staticmethod
    def make_ols_feature(start, stop, n_groups):
        """
        Construct simple OLS features (1, x) for HierMLR as nd.array.

        Parameters
        ----------
        start:
            Start value for OLS feature.
        stop:
            Stop value for OLS feature.

        n_groups:
            number of groups in the hierarchical model.
        """
        X_flat = MultinomialLogisticRegression.make_ols_feature(start, stop)
        return np.stack([X_flat] * n_groups, axis=-1)

    def augment_data(self, data: dict) -> None:
        T, G = data["N"].shape
        data["tau"] = self.tau
        data["X"] = self.make_ols_feature(0, T, G)

    @staticmethod
    def forecast_frequencies(samples, forecast_L):
        """
        Use posterior beta to forecast posterior frequencies.
        """

        # Making feature matrix for forecasting
        last_T = samples["freq"].shape[1]
        n_groups = samples["freq"].shape[-1]

        X = HierMLR.make_ols_feature(
            start=last_T, stop=last_T + forecast_L, n_groups=n_groups
        )

        # (T, F, G) times (F, V, G) -> (T, V, G)
        dot_by_group = vmap(jnp.dot, in_axes=(-1, -1), out_axes=-1)
        dbg_by_sample = vmap(dot_by_group, in_axes=(None, 0), out_axes=0)

        # Creating frequencies from posterior beta
        beta = jnp.array(samples["beta"])
        logits = dbg_by_sample(X, beta)
        samples["freq_forecast"] = softmax(logits, axis=-2)  # (S, T, V, G)
        return samples
