from typing import List, Optional
from jax._src.nn.functions import softmax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax

from evofr.models.renewal_model.basis_functions.basis_fns import BasisFunction
from .LAS import LaplaceRandomWalk
import numpyro
import numpyro.distributions as dist
from .model_functions import reporting_to_vec
from .model_options import NegBinomCases, DirMultinomialSeq

from evofr.models.model_spec import ModelSpec
from .basis_functions import Spline


def renewal_regression_model_factory(
    g_rev,
    CaseLik=None,
    SeqLik=None,
):
    if CaseLik is None:
        CaseLik = NegBinomCases()
    if SeqLik is None:
        SeqLik = DirMultinomialSeq()

    def _variant_model(cases, seq_counts, N, X, var_names=None, pred=False):
        _, N_variant = seq_counts.shape
        T, k = X.shape

        # Make each variant have a growth advantage defined by a spline.
        # Might want to add control for the strength of gam_delta
        with numpyro.plate("N_variant_m1", N_variant - 1):
            delta_0 = numpyro.sample("delta_0", dist.Normal(0.0, 0.5))
            gam_delta = numpyro.sample("gam_delta", dist.HalfNormal(0.01))
            delta_rw = numpyro.sample(
                "delta_rw", LaplaceRandomWalk(scale=gam_delta, num_steps=k)
            )
            delta = delta_0 + delta_rw.T
        delta_mat = jnp.hstack((delta, jnp.zeros((k, 1))))
        freq = numpyro.deterministic(
            "freq", softmax(jnp.cumsum(jnp.dot(X, delta_mat), axis=0), axis=-1)
        )

        # Make total incidence defined by spline
        gam = numpyro.sample("gam", dist.HalfCauchy(0.1))

        beta_rw = numpyro.sample(
            "beta_rw", dist.GaussianRandomWalk(scale=gam, num_steps=k - 1)
        )
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 10.0))
        beta = numpyro.deterministic(
            "beta", beta_0 + jnp.concatenate([jnp.array([0.0]), beta_rw])
        )

        total_incidence = numpyro.deterministic(
            "total_smooth_prev", jnp.exp(jnp.dot(X, beta))
        )
        incidence = freq * total_incidence[:, None]

        # Compute reporting rate
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5.0, 5.0))
        rho_vec = reporting_to_vec(rho, T)
        numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * incidence)

        # Evaluate case likelihood
        CaseLik.model(cases, rho_vec * total_incidence, pred=pred)

        # Evaluate frequency likelihood
        SeqLik.model(seq_counts, N, freq, pred=pred)

        # Compute R and ga based on incidence above using generation time
        Rt, ga = rt_from_incidence(incidence, g_rev, T)

        numpyro.deterministic("R", Rt)
        numpyro.deterministic("ga", ga)
        numpyro.deterministic(
            "r", jnp.diff(jnp.log(incidence), prepend=0.0, axis=0)
        )

    return _variant_model


def rt_from_incidence(incidence, gen_rev, T):
    N_variants = incidence.shape[-1]
    max_age = gen_rev.shape[0]

    # Get indices of interest
    times = np.arange(T)
    ages = np.arange(max_age)
    gen_interval = ages + times[:, None]

    # Compute Rt
    inc_padded = jnp.vstack((jnp.zeros((max_age, N_variants)), incidence))
    infectivity = jnp.einsum("d, tdv -> tv", gen_rev, inc_padded[gen_interval])

    Rt = jnp.divide(incidence, infectivity)
    ga = jnp.divide(Rt, Rt[:, -1][:, None])
    return Rt, ga


class RenewalRegressionModel(ModelSpec):
    def __init__(
        self,
        gen,
        k: Optional[int] = None,
        CLik=None,
        SLik=None,
        v_names: Optional[List[str]] = None,
        basis_fn: Optional[BasisFunction] = None,
    ):
        self.gen = gen
        self.v_names = v_names

        # Making basis expansion for Rt
        self.k = k if k else 10
        self.basis_fn = (
            basis_fn if basis_fn else Spline(s=None, order=4, k=self.k)
        )

        self.CLik = CLik
        self.SLik = SLik
        self.make_model()

    def make_model(self):
        self.model_fn = renewal_regression_model_factory(
            jnp.flip(self.gen, axis=-1),
            self.CLik,
            self.SLik,
        )

    def augment_data(self, data, order=4):
        T = len(data["cases"])
        s = jnp.linspace(0, T, self.k)
        data["X"] = Spline.matrix(jnp.arange(T), s, order=order)
