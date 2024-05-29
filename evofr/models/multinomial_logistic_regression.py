from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import vmap
from jax.nn import softmax

from .model_spec import ModelSpec
from .renewal_model.model_options import DirMultinomialSeq


def simulate_MLR_freq(growth_advantage, freq0, tau, max_time):
    times = np.arange(max_time)
    delta = np.log(growth_advantage) / tau  # to relative fitness
    ufreq = freq0 * np.exp(delta * times[..., None])
    return ufreq / ufreq.sum(axis=-1)[..., None]


def simulate_MLR(growth_advantage, freq0, tau, Ns):
    max_time = len(Ns)
    freq = simulate_MLR_freq(growth_advantage, freq0, tau, max_time)
    seq_counts = [np.random.multinomial(Ns[t], freq[t, :]) for t in range(max_time)]
    return freq, np.stack(seq_counts)


def MLR_numpyro(
    seq_counts,
    N,
    X,
    tau=None,
    pred=False,
    var_names=None,
    dir_multinomial=False,
    xi_prior=None,
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
    freq = numpyro.deterministic("freq", softmax(logits, axis=-1))

    # Evaluate likelihood using either Multinomial or Dirichlet-multinomial
    if dir_multinomial:
        seq_likelihood = DirMultinomialSeq(xi_prior=xi_prior)
        seq_likelihood.model(seq_counts, N, freq, pred=pred)
    else:
        obs = None if pred else np.nan_to_num(seq_counts)
        numpyro.sample(
            "seq_counts",
            dist.MultinomialLogits(logits=logits, total_count=np.nan_to_num(N)),
            obs=obs,
        )

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(beta[-1, :-1] * tau)
        )  # Last row corresponds to linear predictor / growth advantage


class MultinomialLogisticRegression(ModelSpec):
    def __init__(
        self,
        tau: float,
        dir_multinomial=False,
        xi_prior: Optional[float] = None,
    ) -> None:
        """Construct ModelSpec for Multinomial logistic regression

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.
        dir_multinomial:
            Whether to use a Dirichlet-Multinomial likelihood for sequence counts.
            Default is False.
        xi_prior:
            Prior on seq_counts overdispersion magnitude only used when dir_multinomial == True.

        Returns
        -------
        MultinomialLogisticRegression
        """
        self.tau = tau  # Fixed generation time
        self.dir_multinomial = dir_multinomial
        self.xi_prior = xi_prior
        self.model_fn = partial(
            MLR_numpyro, dir_multinomial=self.dir_multinomial, xi_prior=self.xi_prior
        )

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
        data["X"] = self.make_ols_feature(0, T)  # Use intercept and time as predictors

    @staticmethod
    def forecast_frequencies(samples, forecast_L):
        """
        Use posterior beta to forecast posterior frequenicies.
        """

        # Making feature matrix for forecasting
        last_T = samples["freq"].shape[1]
        X = MultinomialLogisticRegression.make_ols_feature(
            start=last_T, stop=last_T + forecast_L
        )

        # Posterior beta
        beta = jnp.array(samples["beta"])

        # Matrix multiplication by sample
        dot_by_sample = vmap(jnp.dot, in_axes=(None, 0), out_axes=0)
        logits = dot_by_sample(X, beta)  # Logit frequencies by variant
        samples["freq_forecast"] = softmax(logits, axis=-1)
        return samples
