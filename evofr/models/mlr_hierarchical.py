import numpy as np
from jax import vmap
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist

from .model_spec import ModelSpec
from .multinomial_logistic_regression import MultinomialLogisticRegression


def hier_MLR_numpyro(seq_counts, N, X, tau=None, pred=False, var_names=None):
    _, N_variants, N_groups = seq_counts.shape
    _, N_features, _ = X.shape

    # Sampling parameters
    with numpyro.plate("features", N_features, dim=-3):
        with numpyro.plate("variants", N_variants - 1, dim=-2):
            # Define loc and scale for beta for predictor and variant group
            beta_loc = numpyro.sample("beta_loc", dist.Normal(0.0, 3.0))
            beta_scale = numpyro.sample("beta_scale", dist.HalfNormal(1.0))

            # Use location and scale parameter to draw within group
            with numpyro.plate("group", N_groups, dim=-1):
                raw_beta = numpyro.sample(
                    "raw_beta", dist.Normal(beta_loc, beta_scale)
                )

    # All parameters are relative to last column / variant
    beta = numpyro.deterministic(
        "beta",
        jnp.concatenate(
            (raw_beta, jnp.zeros((N_features, 1, N_groups))), axis=1
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


class HierMLR(ModelSpec):
    def __init__(self, tau: float) -> None:
        """Construct ModelSpec for Hierarchial multinomial logistic regression.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        Returns
        -------
        HierMLR
        """
        self.tau = tau  # Fixed generation time
        self.model_fn = hier_MLR_numpyro

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
