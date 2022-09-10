from .model_spec import ModelSpec
import numpyro
import numpyro.distributions as dist
import numpy as np
import jax.numpy as jnp
from jax import jit, lax


def compute_frequency(ga, q0, gen_rev, T):
    _ga = jnp.append(ga, 1.0)
    max_age = gen_rev.shape[-1]
    N_variants = q0.shape[-1]

    q0_padded = jnp.vstack((jnp.zeros((max_age - 1, N_variants)), q0))

    @jit
    def _scan_frequency(q, xs):
        # Compute weighted frequency
        q_mag = _ga * jnp.einsum("dv, d -> v", q, gen_rev)
        q_new = q_mag / q_mag.sum()  # Renormalize
        return jnp.vstack((q[-(max_age - 1) :, :], q_new)), q_new[:, None]

    _, q = lax.scan(_scan_frequency, init=q0_padded, length=T - 1, xs=None)
    return jnp.vstack((q0, jnp.squeeze(q)))


def Ito_model_numpyro(seq_counts, N, gen_rev, var_names=None, pred=False):
    T, N_variants = seq_counts.shape

    # Intial frequency
    q0 = numpyro.sample(
        "q0", dist.Dirichlet(jnp.ones(N_variants) / N_variants)
    )

    # Growth advantages
    with numpyro.plate("growth_advantage", N_variants - 1):
        ga = numpyro.sample("ga", dist.LogNormal(loc=0.0, scale=1.0))

    freq = numpyro.deterministic("freq", compute_frequency(ga, q0, gen_rev, T))
    numpyro.deterministic("s", ga - 1)

    # Compute likelihood
    obs = None if pred else np.nan_to_num(seq_counts)
    numpyro.sample(
        "seq_counts",
        dist.Multinomial(probs=freq, total_count=np.nan_to_num(N)),
        obs=obs,
    )


class PianthamModel(ModelSpec):
    """
    Model of frequency dynamics from Piantham 2021
    'Estimating the elevated transmissibility of the B.1.1.7
    strain over previously circulating strains in England
    using GISAID sequence frequencies'.
    """

    def __init__(self, gen):
        """Construct ModelSpec for frequency model with non-trivial generation time.

        Parameters
        ----------
        gen:
            Assumed generation time.

        Returns
        -------
        MLRNowcast
        """
        self.gen = gen
        self.model_fn = Ito_model_numpyro

    def augment_data(self, data: dict) -> None:
        data["gen_rev"] = jnp.flip(self.gen, axis=-1)
        return None

#TODO: Generate Rt based on equation (3) in paper?
