import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .LAS import LaplaceRandomWalk
from .model_helpers import is_obs_idx, pad_to_obs


class FixedGA:
    def __init__(self, gam_prior=0.5, prior_family="Cauchy"):
        """Construct FixedGrowthAdvantage model.
        Parameters
        ----------
        gam_prior:
            hyper_prior for scale of Rt increment.
        prior_family:
            family to draw scale from for LAS.
        Returns
        -------
        FixedGA
        """
        self.gam_prior = gam_prior

        if prior_family == "Cauchy":
            self.scale_prior = dist.HalfCauchy()
        elif prior_family == "Normal":
            self.scale_prior = dist.HalfNormal()

    def model(self, N_variant, X):
        _, k = X.shape

        # Locally adaptive smoothing
        gam = numpyro.sample("gam", self.scale_prior) * self.gam_prior
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 1.0))
        with numpyro.plate("N_steps_base", k - 1):
            beta_rw_step = numpyro.sample("beta_rw_step", dist.Laplace()) * gam
            beta_rw = numpyro.deterministic(
                "beta_rw", jnp.cumsum(beta_rw_step)
            )

        # Combine increments and starting position
        beta_rw = jnp.append(jnp.zeros(1), beta_rw)
        beta = beta_0 + beta_rw

        # Getting log variant growth advantages
        with numpyro.plate("N_variant_m1", N_variant - 1):
            v = numpyro.sample("v", dist.Normal(0.0, 0.5))

        numpyro.deterministic(
            "ga", jnp.exp(v)
        )  # Transform to growth advantage

        # Computing R
        R = numpyro.deterministic(
            "R", jnp.exp((X @ beta + jnp.append(v, 0.0)[:, None])).T
        )
        return R


class FreeGrowth:
    def __init__(self, gam_prior=0.5, prior_family="Cauchy"):
        self.gam_prior = gam_prior

        if prior_family == "Cauchy":
            self.scale_prior = dist.HalfCauchy()
        elif prior_family == "Normal":
            self.scale_prior = dist.HalfNormal()

    def model(self, N_variant, X):
        _, k = X.shape

        # Locally adaptive smoothing on all R trajectories
        with numpyro.plate("variant_beta", N_variant):
            gam = numpyro.sample("gam", self.scale_prior) * self.gam_prior
            beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 0.5))
            with numpyro.plate("N_steps_base", k - 1):
                beta_rw_step = (
                    numpyro.sample("beta_rw_step", dist.Laplace()) * gam
                )
                beta_rw = numpyro.deterministic(
                    "beta_rw", jnp.cumsum(beta_rw_step, axis=0)
                )

        beta_rw = jnp.row_stack((jnp.zeros((1, N_variant)), beta_rw))
        beta = beta_0 + beta_rw

        # Computing R
        R = numpyro.deterministic("R", jnp.exp(X @ beta))
        return R


class GARW:
    def __init__(
        self, gam_prior=0.5, gam_delta_prior=0.5, prior_family="Cauchy"
    ):
        """Construct FixedGrowthAdvantage model.
        Parameters
        ----------
        gam_prior:
            hyper_prior for scale of Rt increment.
        gam_delta_prior:
            hyper_prior for scale of growth advantage increment.
        prior_family:
            family to draw scale from for LAS.

        Returns
        -------
        GARW
        """

        self.gam_prior = gam_prior
        self.gam_delta_prior = gam_delta_prior

        if prior_family == "Cauchy":
            self.scale_prior = dist.HalfCauchy()
        elif prior_family == "Normal":
            self.scale_prior = dist.HalfNormal()

    def model(self, N_variant, X):
        _, k = X.shape

        # Time varying base trajectory
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 1.0)) * 0.2

        # Simulate increments for base trajectory
        gam = numpyro.sample("gam", self.scale_prior) * self.gam_prior
        # beta_rw = numpyro.sample(
        #     "beta_rw", LaplaceRandomWalk(scale=gam, num_steps=k - 1)
        # )
        with numpyro.plate("N_steps_base", k - 1):
            beta_rw_step = numpyro.sample("beta_rw_step", dist.Laplace()) * gam
            beta_rw = numpyro.deterministic(
                "beta_rw", jnp.cumsum(beta_rw_step)
            )

        # Combine increments and starting position
        beta_rw = jnp.append(jnp.zeros(1), beta_rw)
        beta = beta_0 + beta_rw

        # Time varying growth advantage as random walk
        # Regularizes changes in growth advantage of variants
        with numpyro.plate("N_variant_m1", N_variant - 1):
            delta_0 = numpyro.sample("delta_0", dist.Normal(0.0, 1.0)) * 0.2
            gam_delta = (
                numpyro.sample("gam_delta", self.scale_prior)
                * self.gam_delta_prior
            )
            # delta_rw = numpyro.sample(
            #     "delta_rw", LaplaceRandomWalk(scale=gam_delta, num_steps=k - 1)
            # )

            with numpyro.plate("N_steps_variant", k - 1):
                delta_rw_step = (
                    numpyro.sample("delta_rw_step", dist.Laplace()) * gam_delta
                )
                delta_rw = numpyro.deterministic(
                    "delta_rw", jnp.cumsum(delta_rw_step, axis=0)
                )

            delta_rw = jnp.concatenate(
                (jnp.zeros((1, N_variant - 1)), delta_rw), axis=0
            )
            delta = delta_0 + delta_rw

        # Transform to growth advatnage
        numpyro.deterministic("ga", jnp.exp(X @ delta))

        # Construct beta matrix
        beta_mat = beta[:, None] + jnp.hstack((delta, jnp.zeros((k, 1))))
        R = numpyro.deterministic("R", jnp.exp((X @ beta_mat)))
        return R


class GAPRW:
    def __init__(self, gam_prior=0.5, gam_delta_prior=0.5):
        self.gam_prior = gam_prior
        self.gam_delta_prior = gam_delta_prior

    def model(self, N_variant, X):
        T, _ = X.shape

        # Time varying base trajectory
        gam = numpyro.sample("gam", dist.HalfNormal()) * self.gam_prior
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 1.0))
        with numpyro.plate("N_steps_base", T - 1):
            beta_rw_step = numpyro.sample("beta_rw_step", dist.Laplace()) * gam
            beta_rw = numpyro.deterministic(
                "beta_rw", jnp.cumsum(beta_rw_step)
            )

        # Combine increments and starting position
        beta_rw = jnp.append(jnp.zeros(1), beta_rw)
        beta = beta_0 + beta_rw

        # Time varying growth advantage as random walk
        # Regularizes changes in growth advantage of variants
        with numpyro.plate("N_variant_m1", N_variant - 1):
            delta_0 = numpyro.sample("delta_0", dist.Normal(0.0, 0.5))
            gam_delta = numpyro.sample(
                "gam_delta", dist.HalfNormal(self.gam_delta_prior)
            )
            # delta_rw = numpyro.sample(
            #     "delta_rw", LaplaceRandomWalk(scale=gam_delta, num_steps=T - 1)
            # )
            # delta_rw = jnp.concatenate(
            #     (jnp.zeros((N_variant - 1, 1)), delta_rw), axis=-1
            # )
            #
            # delta = delta_0 + delta_rw.T
            with numpyro.plate("N_steps_variant", T - 1):
                delta_rw_step = (
                    numpyro.sample("delta_rw_step", dist.Laplace()) * gam_delta
                )
                delta_rw = numpyro.deterministic(
                    "delta_rw", jnp.cumsum(delta_rw_step, axis=0)
                )

            delta_rw = jnp.concatenate(
                (jnp.zeros((1, N_variant - 1)), delta_rw), axis=0
            )
            delta = delta_0 + delta_rw

        # Transform to growth advatnage
        numpyro.deterministic("ga", jnp.exp(delta))

        # Construct beta matrix
        beta_mat = beta[:, None] + jnp.hstack((delta, jnp.zeros((T, 1))))
        R = numpyro.deterministic("R", jnp.exp(beta_mat))
        return R


class PoisCases:
    def __init__(self):
        pass

    def model(self, cases, EC, pred=False):
        is_obs = is_obs_idx(cases)  # Find unobserved case counts
        # Overwrite defaults for predictive checks
        obs = None if pred else np.nan_to_num(cases)

        numpyro.sample("cases", dist.Poisson(pad_to_obs(EC, is_obs)), obs=obs)


class ZIPoisCases:
    def __init__(self):
        pass

    def model(self, cases, EC, pred=False):

        # Finding zero locations and making only parameters for zero observation
        is_zero = cases == 0
        zero_idx = np.nonzero(is_zero)[0]

        gate = jnp.zeros_like(cases) + 1e-12
        if zero_idx.shape[0] > 0:
            with numpyro.plate("n_zero", zero_idx.shape[0]):
                zp = numpyro.sample("zp", dist.Beta(0.1, 0.1))
            gate = gate.at[zero_idx].set(zp)

        # Overwrite defaults for predictive checks
        gate = jnp.zeros_like(cases) if pred else gate
        obs = None if pred else np.nan_to_num(cases)

        numpyro.sample(
            "cases", dist.ZeroInflatedPoisson(rate=EC, gate=gate), obs=obs
        )


class NegBinomCases:
    def __init__(self, raw_alpha_sd=0.01):
        self.raw_alpha_sd = raw_alpha_sd

    def model(self, cases, EC, pred=False):
        # NegativeBinomial sampling
        raw_alpha = numpyro.sample(
            "raw_alpha", dist.HalfNormal(self.raw_alpha_sd)
        )
        is_obs = is_obs_idx(cases)  # Find unobserved case counts

        # Overwrite defaults for predictive checks
        obs = None if pred else np.nan_to_num(cases)

        numpyro.sample(
            "cases",
            dist.NegativeBinomial2(
                mean=pad_to_obs(EC, is_obs),
                concentration=jnp.power(raw_alpha, -2),
            ),
            obs=obs,
        )


class ZINegBinomCases:
    def __init__(self, raw_alpha_sd=0.01):
        self.raw_alpha_sd = raw_alpha_sd

    def model(self, cases, EC, pred=False):
        # NegativeBinomial sampling
        raw_alpha = (
            numpyro.sample("raw_alpha", dist.HalfNormal()) * self.raw_alpha_sd
        )

        # Finding zero locations and making only parameters for zero observation
        is_zero = cases == 0
        zero_idx = np.nonzero(is_zero)[0]

        gate = jnp.zeros_like(cases) + 1e-12
        if zero_idx.shape[0] > 0:
            with numpyro.plate("n_zero", zero_idx.shape[0]):
                zp = numpyro.sample("zp", dist.Beta(0.1, 0.1))
            gate = gate.at[zero_idx].set(zp)

        # Overwrite defaults for predictive checks
        gate = jnp.zeros_like(cases) if pred else gate
        obs = None if pred else np.nan_to_num(cases)

        numpyro.sample(
            "cases",
            dist.ZeroInflatedNegativeBinomial2(
                mean=EC, gate=gate, concentration=jnp.power(raw_alpha, -2)
            ),
            obs=obs,
        )


class MultinomialSeq:
    def __init__(self):
        pass

    def model(self, seq_counts, N, freq, pred=False):

        # Sample with Multinomial
        obs = None if pred else np.nan_to_num(seq_counts)
        numpyro.sample(
            "seq_counts",
            dist.Multinomial(total_count=np.nan_to_num(N), probs=freq),
            obs=obs,
        )


class DirMultinomialSeq:
    def __init__(self, xi_prior=None):
        self.xi_prior = xi_prior if xi_prior is not None else 99

    def model(self, seq_counts, N, freq, pred=False):
        # Overdispersion in sequence counts
        xi = numpyro.sample("xi", dist.Beta(1, self.xi_prior))
        trans_xi = jnp.reciprocal(xi) - 1
        # xi = numpyro.sample("xi", dist.HalfNormal())
        # trans_xi = self.xi_prior / xi

        obs = None if pred else np.nan_to_num(seq_counts)

        # Sample with DirichletMultinomial
        numpyro.sample(
            "seq_counts",
            dist.DirichletMultinomial(
                total_count=np.nan_to_num(N),
                concentration=1e-8 + trans_xi * freq,
            ),
            obs=obs,
        )
