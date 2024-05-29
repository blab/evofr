from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import jit, vmap
from jax._src.interpreters.batching import Array
from jax.nn import softmax
from jax.scipy.special import gammaln
from numpyro.infer.reparam import TransformReparam

from .model_spec import ModelSpec


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


def hier_MLR_hsgp_numpyro(
    seq_counts,
    N,
    hsgp,
    tau=None,
    pool_scale=None,
    pred=False,
    var_names=None,
):
    N_time, N_variants, N_groups = seq_counts.shape

    pool_scale = 0.1 if pool_scale is None else pool_scale

    # Generate features matrix
    phi = hsgp.make_features(np.arange(N_time))
    N_features = phi.shape[-1]

    # Sample HSGP parameters
    spd = numpyro.deterministic("sqrt_spd", jnp.sqrt(hsgp.model()))

    # Sampling intercept and fitness parameters
    reparam_config = {
        "beta_loc": TransformReparam(),
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
                # Define loc and scale for feature beta
                beta_loc = numpyro.sample(
                    "beta_loc",
                    dist.TransformedDistribution(
                        dist.Normal(0.0, 1.0),
                        dist.transforms.AffineTransform(0.0, 0.2),
                    ),
                )

                # Use location and scale parameter to draw within group
                with numpyro.plate("group", N_groups, dim=-1):
                    # Leave intercept alpha unpooled
                    raw_beta = numpyro.sample(
                        "raw_beta",
                        dist.TransformedDistribution(
                            dist.Normal(0.0, 1.0),
                            dist.transforms.AffineTransform(beta_loc, beta_scale),
                        ),
                    )

    with numpyro.plate("group_", N_groups):
        with numpyro.plate("variants_", N_variants - 1):
            _init_logit = numpyro.sample("init_logit", dist.Normal(0, 6.0))
            intercept = numpyro.sample("intercept", dist.Normal(0, 1.0))

    # All parameters are relative to last column / variant
    beta = numpyro.deterministic(
        "beta",
        raw_beta,
    )
    # Compute fitness from beta and phi
    # (T, F, G) times (F, V, G) -> (T, V, G)
    _fitness = numpyro.deterministic(
        "delta",
        jnp.einsum("tj, jvg -> tvg", phi, spd[..., None, None] * beta)
        + intercept[None, :, :],
    )
    fitness = jnp.hstack((_fitness, jnp.zeros((N_time, 1, N_groups))))

    # Combine initial logits and fitness
    init_logit = jnp.vstack((_init_logit, jnp.zeros((1, N_groups))))
    logits = jnp.cumsum(fitness.at[0, :].set(0), axis=0) + init_logit

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
    seq_counts = numpyro.deterministic("seq_counts", jnp.swapaxes(_seq_counts, 2, 1))

    # Compute frequency
    numpyro.deterministic("freq", softmax(logits, axis=1))

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(fitness * tau)
        )  # Last row corresponds to linear predictor / growth advantage
        # delta_loc = numpyro.deterministic(
        #   "delta",
        #   jnp.einsum("tj, jvg -> tvg", phi, spd[..., None, None] * beta)
        #   + intercept[None, :, :],
        # )
        # numpyro.deterministic("ga_loc", jnp.exp(delta_loc * tau))
    return None


class HierMLR_HSGP(ModelSpec):
    def __init__(
        self,
        tau: float,
        hsgp: HSGaussianProcess,
        pool_scale: Optional[float] = None,
    ) -> None:
        """Construct ModelSpec for Hierarchial multinomial logistic regression.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        hsgp:
            Hilbert space Gaussian process class used to generate relative fitneses,

        pool_scale:
            Prior standard deviation for pooling of growth advantages.

        Returns
        -------
        HierMLRTime
        """
        self.tau = tau  # Fixed generation time
        self.pool_scale = pool_scale  # Prior std for coefficients
        self.hsgp = hsgp

        self.model_fn = partial(
            hier_MLR_hsgp_numpyro,
            pool_scale=self.pool_scale,
            hsgp=hsgp,
        )

    def augment_data(self, data: dict) -> None:
        T, G = data["N"].shape
        data["tau"] = self.tau

    def forecast_frequencies(
        self, samples, forecast_L, tau=None, linear=False, mean_len=7
    ):
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
            forecast_times = jnp.arange(0.0, forecast_L) + 1
        else:  # Keep delta_pred constant
            forecast_times = jnp.ones(forecast_L)
        delta_pred = delta_pred[:, None, :, :] * forecast_times[None, :, None, None]

        # Creating frequencies from posterior beta
        logits = jnp.log(samples["freq"])[:, -1, :, :]
        logits = logits[:, None, :, :] + jnp.cumsum(
            delta_pred, axis=1
        )  # Project based on delta_pred: adding cummulatively in time
        samples["freq_forecast"] = softmax(logits, axis=-2)  # (S, T, V, G)
        if tau is None:
            tau = self.tau
        samples["ga_forecast"] = jnp.exp(delta_pred * tau)
        return samples
