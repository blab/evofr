from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array, jit, vmap
from jax.nn import softmax
from jax.scipy.special import gammaln

import evofr as ef


class HSGaussianProcess:
    """
    Implementation of basis approximation to Gaussian processes.
    This produces basis functions for the Hilbert Space approximate Gaussian process.
    Reference: https://arxiv.org/abs/2004.11408
    """

    def __init__(self, L, num_basis):
        self.L = L
        self.num_basis = num_basis
        self.lams = self.eigenvalues(self.L, jnp.arange(self.num_basis))

    @staticmethod
    def eigenvalues(L: float, j: int):
        return jnp.square(j * jnp.pi / (2 * L))

    @staticmethod
    def phi(L: float, j: int, x: Array):
        lam = HSGaussianProcess.eigenvalues(L, j)
        arg = jnp.sqrt(lam) * (x + L)
        return jnp.sqrt(1 / L) * jnp.sin(arg)

    @staticmethod
    def phi_matrix(L: float, js: Array, x: Array):
        phi_mapped = jit(
            vmap(HSGaussianProcess.phi, in_axes=(None, 0, None), out_axes=-1)
        )  # Map across m values
        return phi_mapped(L, js, x)

    def make_features(self, ts) -> Array:
        # Make eigenvectors
        ms = jnp.arange(1, self.num_basis + 1)
        phi = self.phi_matrix(self.L, ms, ts)
        return phi


def assign_priors(name, val, default: dist.Distribution):
    if val is None:
        return numpyro.sample(name, default)
    elif isinstance(val, dist.Distribution):
        return numpyro.sample(name, val)
    else:
        return val


class SquaredExponential(HSGaussianProcess):
    def __init__(
        self,
        alpha: Optional[any] = None,
        rho: Optional[any] = None,
        L: Optional[float] = None,
        num_basis: Optional[int] = None,
    ):
        """
        HSGP using a squared exponential or radial basis function kernel.

        alpha: marginal variance
        rho: length-scale
        L: ...

        """
        self.alpha = alpha
        self.rho = rho
        super().__init__(L=L if L else 10.0, num_basis=num_basis if num_basis else 50)

    @staticmethod
    def spd(alpha: float, rho: float, w: Array):
        return alpha * jnp.sqrt(2 * jnp.pi) * rho * jnp.exp(-0.5 * jnp.square(rho * w))

    def model(self):
        alpha = assign_priors("alpha", self.alpha, default=dist.HalfNormal(1e-3))
        rho = assign_priors("rho", self.rho, default=dist.HalfNormal(25))
        return self.spd(alpha, rho, jnp.sqrt(self.lams))


class Matern(HSGaussianProcess):
    def __init__(
        self,
        alpha: Optional[any] = None,
        rho: Optional[any] = None,
        nu: Optional[float] = None,
        L: Optional[float] = None,
        num_basis: Optional[int] = None,
    ):
        self.alpha = alpha
        self.rho = rho
        self.nu = nu if nu else 5 / 2
        super().__init__(L=L if L else 10.0, num_basis=num_basis if num_basis else 50)

    @staticmethod
    def spd(alpha: float, rho: float, nu: float, w: Array):
        gammanu = jnp.exp(gammaln(nu))
        gammanuhalf = jnp.exp(gammaln(nu + 0.5))
        coef = (
            alpha
            * 2
            * jnp.sqrt(jnp.pi)
            * gammanuhalf
            * jnp.power(2 * nu, nu)
            / (gammanu * jnp.power(rho, 2 * nu))
        )
        base = 2 * nu * jnp.power(rho, -2) + 4 * jnp.square(jnp.pi * w)
        expon = -nu - 0.5
        return coef * jnp.power(base, expon)

    def model(self):
        alpha = assign_priors("alpha", self.alpha, default=dist.HalfNormal(1e-3))
        rho = assign_priors("rho", self.rho, default=dist.HalfNormal(25))
        return self.spd(alpha, rho, self.nu, jnp.sqrt(self.lams))


class SpectralMixture(HSGaussianProcess):
    def __init__(
        self,
        num_components: int,
        mixture_weights: Optional[Array] = None,
        mixture_means: Optional[Array] = None,
        mixture_sigmas: Optional[Array] = None,
        L: Optional[float] = None,
        num_basis: Optional[int] = None,
    ):
        self.num_components = num_components
        self.mixture_weights = mixture_weights
        self.mixture_means = mixture_means
        self.mixture_sigmas = mixture_sigmas
        super().__init__(L=L if L else 10.0, num_basis=num_basis if num_basis else 50)

    @staticmethod
    def spd(
        mixture_weights: Array, mixture_means: Array, mixture_sigmas: Array, w: Array
    ):
        coefs = jnp.reciprocal(jnp.sqrt(2 * jnp.pi)) / mixture_sigmas
        args = jnp.square((w[..., None] - mixture_means) / mixture_sigmas)
        return (coefs * mixture_weights * jnp.exp(-0.5 * args)).sum(axis=-1)

    def model(self):
        mixture_weights = assign_priors(
            "mixture_weights",
            self.mixture_weights,
            default=dist.Dirichlet(jnp.ones(self.num_components)),
        )
        mixture_means = assign_priors(
            "mixture_means",
            self.mixture_means,
            default=dist.TransformedDistribution(
                dist.Normal(0, 1.0).expand((self.num_components,)),
                dist.transforms.OrderedTransform(),
            ),
        )
        mixture_sigmas = assign_priors(
            "mixture_sigmas",
            self.mixture_sigmas,
            default=dist.HalfNormal(0.5).expand((self.num_components,)),
        )
        return self.spd(
            mixture_weights, mixture_means, mixture_sigmas, jnp.sqrt(self.lams)
        )


def not_yet_observed(seq_counts):
    """
    Compute binary predictor for whether a variant has been seen by time t.
    """
    T, V = seq_counts.shape
    never_seen = np.ones_like(seq_counts)
    for v in range(V):
        for t in range(1, T):
            # If we haven't seen yet, check that we still havent
            if never_seen[t - 1, v]:
                never_seen[t, v] = seq_counts[t, v] == 0
            # If we have seen it, we know it's 0
            else:
                never_seen[t, v] = 0
    return never_seen


def relative_fitness_hsgp_numpyro(
    seq_counts, N, hsgp, tau=None, pred=False, var_names=None
):
    N_time, N_variants = seq_counts.shape

    # Generate features matrix
    phi = hsgp.make_features(np.arange(N_time))
    num_basis = phi.shape[-1]

    # Sample HSGP parameters
    spd = numpyro.deterministic("sqrt_spd", jnp.sqrt(hsgp.model()))

    with numpyro.plate("variant", N_variants - 1):
        _init_logit = numpyro.sample("init_logit", dist.Normal(0, 6.0))
        intercept = numpyro.sample("intercept", dist.Normal(0, 1.0))
        with numpyro.plate("num_basis", num_basis):
            beta = numpyro.sample("beta", dist.Normal(0, 1))
    _fitness = numpyro.deterministic(
        "delta",
        jnp.einsum("tj, jv -> tv", phi, spd[..., None] * beta) + intercept[None, :],
    )
    fitness = jnp.hstack((_fitness, jnp.zeros((N_time, 1))))

    # Sum fitness to get dynamics over time
    init_logit = jnp.append(_init_logit, 0.0)
    logits = jnp.cumsum(fitness.at[0, :].set(0), axis=0) + init_logit

    # Adjust for introductions
    never_seen = not_yet_observed(seq_counts)
    logits = logits - 10 * never_seen

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
            "ga", jnp.exp(fitness[:, :-1] * tau)
        )  # Last row corresponds to linear predictor / growth advantage


class RelativeFitnessHSGP(ef.ModelSpec):
    def __init__(
        self,
        hsgp: Optional[HSGaussianProcess] = None,
        tau: Optional[float] = None,
    ):
        self.hsgp = hsgp if hsgp is not None else SquaredExponential(L=50, num_basis=30)
        self.tau = tau
        self.model_fn = partial(
            relative_fitness_hsgp_numpyro, hsgp=self.hsgp, tau=self.tau
        )

    def augment_data(self, data: dict) -> None:
        return None

    def fit_mcmc(
        self,
        data: ef.VariantFrequencies,
        num_warmup: int = 100,
        num_samples: int = 100,
    ) -> ef.PosteriorHandler:
        """
        Abstract away NUTS stuff in Evofr and numpyro for quick usage.
        """
        inference_method = ef.InferNUTS(num_warmup=num_warmup, num_samples=num_samples)
        return inference_method.fit(self, data)

    def forecast_mcmc(self, samples, forecast_L):
        # Create time points to forecast
        _, N_time, _ = samples["delta"].shape
        pred_ts = np.arange(1, forecast_L + 1) + N_time

        # Create GP features
        phi_forecast = self.hsgp.make_features(pred_ts)

        # Forecast relative fitness
        def _delta_forecast(spd, beta):
            return jnp.einsum("tj, jv -> tv", phi_forecast, spd[..., None] * beta)

        samples["delta_forecast"] = vmap(_delta_forecast, in_axes=(0, 0))(
            samples["sqrt_spd"], samples["beta"]
        )

        # Forecast frequency
        def _forecast_freq(fitness, frequency):
            init_freq = jnp.log(frequency[-1, :])  # Last known frequency
            _fitness = jnp.concatenate((fitness, jnp.zeros((forecast_L, 1))), axis=-1)
            cum_fitness = jnp.cumsum(_fitness.at[0, :].set(0), axis=0)
            return softmax(cum_fitness + init_freq, axis=-1)

        vmap_forecast_freq = jax.vmap(_forecast_freq, in_axes=(0, 0), out_axes=0)
        samples["freq_forecast"] = vmap_forecast_freq(
            samples["delta_forecast"], samples["freq"]
        )
        return samples
