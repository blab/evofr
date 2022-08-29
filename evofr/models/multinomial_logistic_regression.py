import numpy as np
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist

from .model_spec import ModelSpec


def MLR_numpyro(seq_counts, N, X, tau=None, pred=False, var_names=None):
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
        numpyro.deterministic(
            "ga", jnp.exp(beta[-1, :-1] * tau)
        )  # Last row corresponds to linear predictor / growth advantage


class MultinomialLogisticRegression(ModelSpec):
    def __init__(self, tau: float) -> None:
        """Construct ModelSpec for Multinomial logistic regression

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        Returns
        -------
        MultinomialLogisticRegression
        """
        self.tau = tau  # Fixed generation time
        self.model_fn = MLR_numpyro

    @staticmethod
    def make_ols_feature(start, stop):
        """
        Construct simple OLS features (1, x) for MultinomialLogisticRegression.

        Parameters
        ----------
        start:
            Start value for OLS feature.
        stop:
            Stop value for OLS feature.
        """
        t = jnp.arange(start=start, stop=stop)
        return jnp.column_stack((jnp.ones_like(t), t))

    def augment_data(self, data: dict) -> None:
        T = len(data["N"])
        data["tau"] = self.tau
        data["X"] = self.make_ols_feature(
            0, T
        )  # Use intercept and time as predictors
