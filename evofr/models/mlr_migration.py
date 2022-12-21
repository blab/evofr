from functools import partial
from typing import Optional
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap

import numpyro
import numpyro.distributions as dist

from .model_spec import ModelSpec


def nD_drift(freq, beta):
    return jnp.dot(beta[:, None] - beta, freq) * freq


def nD_mig_alt(freq, M):
    # mig[l] = sum(m_[k -> l} x[v,k])  - sum_k(m_{k->l}) * x[v, l]
    imports = jnp.einsum(
        "kl, vk -> vl", M, freq
    )  # Sum across possible sources
    exports = jnp.sum(M, axis=1) * freq  # Sum across possible sinks
    return imports - exports


def nD_mig(freq, M):
    # This progresses frequencies according to mean-reverting model
    # direction[v, k, l] = (x[v,k] - x[v,l])
    # M[k, l] = M_{k,l} i.e. k -> l
    direction = freq[:, None] - freq[None, :]
    return (direction * M).sum(axis=0)  # Sum across possible k


def MLR_mig_sim(beta, freq0, M, num_steps):
    mapped_drift = vmap(
        nD_drift, in_axes=(-1, -1), out_axes=-1
    )  # Location is last column
    mapped_mig = vmap(nD_mig, in_axes=(0, None), out_axes=0)

    @jit
    def _migration_step(freq, xs):
        # Need to compute drift for each set of freq and beta
        freq_next = freq + mapped_drift(freq, beta) + mapped_mig(freq, M)
        freq_next = jnp.clip(freq_next, 1e-16, 1.0 - 1e-16)
        return freq_next, freq_next

    _, freq = lax.scan(_migration_step, xs=None, init=freq0, length=num_steps)
    return jnp.vstack((freq0[None, :, :], freq))


def fill_mig_matrix(non_diag_elems, L):
    non_diag_indices = ~np.identity(L).astype(bool)
    M = jnp.zeros(shape=(L, L))
    return M.at[non_diag_indices].set(non_diag_elems)


def MLR_migration_numpyro(
    seq_counts, N, tau=None, pred=False, var_names=None, pool_scale=None
):
    T, N_variants, L = seq_counts.shape

    pool_scale = 4.0 if pool_scale is None else pool_scale

    with numpyro.plate("variants", N_variants - 1, dim=-2):
        # Define loc and scale for beta for predictor and variant group
        beta_loc = numpyro.sample("beta_loc", dist.Normal(0.0, 5.0))
        beta_scale = numpyro.sample("beta_scale", dist.HalfNormal(pool_scale))

        # Use location and scale parameter to draw within group
        with numpyro.plate("group", L, dim=-1):
            raw_beta = numpyro.sample(
                "raw_beta", dist.Normal(beta_loc, beta_scale)
            )

    # All parameters are relative to last column / variant
    beta = numpyro.deterministic(
        "beta",
        jnp.concatenate((raw_beta, jnp.zeros((1, L))), axis=0),
    )

    # Migration rates and matrix
    M_raw = numpyro.sample(
        "M_raw", dist.Exponential(rate=1), sample_shape=(L * (L - 1),)
    )
    M = numpyro.deterministic("M", fill_mig_matrix(1e-4 * M_raw, L))

    # Initialize frequencies
    with numpyro.plate("group", L, dim=-1):
        freq0 = numpyro.sample("freq0", dist.Dirichlet(jnp.ones(N_variants)))

    # Forward simulate
    freq = numpyro.deterministic("freq", MLR_mig_sim(beta, freq0.T, M, T - 1))

    # Evaluate likelihood
    obs = None if pred else np.swapaxes(np.nan_to_num(seq_counts), 1, 2)

    _seq_counts = numpyro.sample(
        "_seq_counts",
        dist.Multinomial(
            probs=jnp.swapaxes(freq, 1, 2), total_count=np.nan_to_num(N)
        ),
        obs=obs,
    )

    seq_counts = numpyro.deterministic(
        "seq_counts", jnp.swapaxes(_seq_counts, 2, 1)
    )

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(beta[:-1, :] * tau)  # Adapt for multiple locations
        )  # Last row corresponds to linear predictor / growth advantage


class MLR_Migration(ModelSpec):
    def __init__(self, tau: float, pool_scale: Optional[float] = None) -> None:
        """Construct ModelSpec for MLR_Migration.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        Returns
        -------
        MLR_Migration
        """
        self.tau = tau  # Fixed generation time
        self.pool_scale = pool_scale  # Prior std for coefficients
        self.model_fn = partial(
            MLR_migration_numpyro, pool_scale=self.pool_scale
        )

    def augment_data(self, data: dict) -> None:
        T = len(data["N"])
        data["tau"] = self.tau
