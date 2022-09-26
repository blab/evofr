from functools import partial
from evofr.models.renewal_model.model_options import MultinomialSeq
from .model_spec import ModelSpec
import numpyro
import numpyro.distributions as dist
import numpy as np
import jax.numpy as jnp
from jax import jit, lax


def compute_frequency_piantham(ga, q0, gen_rev, T):
    """
    Compute variant frequencies according to Piantham model.

    Parameters
    ----------
    ga:
        Growth advantages for non-baseline variants.

    q0:
        Initial variant frequencies.

    gen_rev:
        Reversed generation time.

    T:
        Total length of time to simulate frequencies for.

    Returns
    -------
    Simulated frequencies as DeviceArray.
    """
    _ga = jnp.append(ga, 1.0)
    max_age = gen_rev.shape[-1]
    N_variants = q0.shape[-1]

    q0_padded = jnp.vstack((jnp.zeros((max_age - 1, N_variants)), q0))

    @jit
    def _scan_frequency(q, _):
        # Compute weighted frequency
        q_mag = _ga * jnp.einsum("dv, d -> v", q, gen_rev)
        q_new = q_mag / q_mag.sum()  # Normalize
        return jnp.vstack((q[-(max_age - 1) :, :], q_new)), q_new[:, None]

    _, q = lax.scan(_scan_frequency, init=q0_padded, length=T - 1, xs=None)
    return jnp.vstack((q0, jnp.squeeze(q)))


def Piantham_model_numpyro(
    seq_counts, N, gen_rev, SeqLik, forecast_L, pred=False
):
    T, N_variants = seq_counts.shape

    # Intial frequency
    q0 = numpyro.sample(
        "q0", dist.Dirichlet(jnp.ones(N_variants) / N_variants)
    )

    # Growth advantages
    with numpyro.plate("growth_advantage", N_variants - 1):
        ga = numpyro.sample("ga", dist.LogNormal(loc=0.0, scale=1.0))

    _T = T + forecast_L if pred else T
    _freq = compute_frequency_piantham(ga, q0, gen_rev, _T)

    freq = numpyro.deterministic(
        "freq", jnp.take(_freq, jnp.arange(T), axis=0)
    )
    numpyro.deterministic("s", ga - 1)

    # Compute likelihood of frequency
    SeqLik.model(seq_counts, N, freq, pred)

    if pred:
        numpyro.deterministic("freq_forecast", _freq[T:, :])


class PianthamModel(ModelSpec):
    """
    Model of frequency dynamics from Piantham 2021
    'Estimating the elevated transmissibility of the B.1.1.7
    strain over previously circulating strains in England
    using GISAID sequence frequencies'.
    """

    def __init__(self, gen, SeqLik=None, forecast_L=None):
        """Construct ModelSpec for frequency model with non-trivial generation time.

        Parameters
        ----------
        gen:
            Assumed generation time.

        SeqLik:
            Optional sequence likelihood option: MultinomialSeq or
            DirMultinomialSeq. Defaults to MultinomialSeq.

        forecast_L:
            Optional forecast length.

        Returns
        -------
        PianthamModel
        """
        self.gen = gen
        self.SeqLik = MultinomialSeq() if SeqLik is None else SeqLik
        self.forecast_L = 0 if forecast_L is None else forecast_L
        self.model_fn = partial(
            Piantham_model_numpyro,
            SeqLik=self.SeqLik,
            forecast_L=self.forecast_L,
        )

    def augment_data(self, data: dict) -> None:
        data["gen_rev"] = jnp.flip(self.gen, axis=-1)

        # Remove unnecessary key from VariantFrequencies
        data.pop("var_names", None)
        return None
