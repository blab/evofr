from typing import List, Optional
import jax.numpy as jnp
import numpy as np

from evofr.models.model_spec import ModelSpec
from evofr.models.renewal_model.model_helpers import to_survivor_function
from .basis_functions import BasisFunction, Spline

from .LAS import LaplaceRandomWalk
from .model_functions import forward_simulate_I_and_prev, reporting_to_vec
from .model_options import NegBinomCases

import numpyro
import numpyro.distributions as dist


def _single_renewal_factory(
    g_rev,
    delays,
    inf_period,
    seed_L,
    forecast_L,
    CaseLik=None,
):
    if CaseLik is None:
        CaseLik = NegBinomCases()

    def _model(cases, X, pred=False):
        T, k = X.shape
        obs_range = jnp.arange(seed_L, seed_L + T, 1)

        # Effective Reproduction number likelihood
        gam = numpyro.sample("gam", dist.HalfNormal(1.0))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 1.0))
        beta_rw = numpyro.sample(
            "beta_rw", LaplaceRandomWalk(scale=gam, num_steps=k)
        )
        beta = beta_0 + beta_rw
        _R = numpyro.deterministic("R", jnp.exp(X @ beta))

        # Add forecasted values of R
        R = _R
        if forecast_L > 0:
            R_forecast = numpyro.deterministic(
                "R_forecast", jnp.vstack((_R[-1],) * forecast_L)
            )
            R = jnp.hstack((_R, R_forecast))

        # Getting initial conditions
        I0 = numpyro.sample("I0", dist.LogNormal(0.0, 5.0))
        intros = jnp.zeros((T + seed_L + forecast_L,))
        intros = intros.at[np.arange(seed_L)].set(I0 * jnp.ones(seed_L))

        # Generate day-of-week reporting fraction
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5.0, 5.0))
        rho_vec = reporting_to_vec(rho, T)

        I_prev, prev = forward_simulate_I_and_prev(
            intros, R, g_rev, delays, inf_period, seed_L
        )
        I_prev = jnp.clip(I_prev, a_min=1e-12, a_max=1e25)

        # Smooth trajectory for plotting
        numpyro.deterministic(
            "I_smooth", jnp.mean(rho_vec) * jnp.take(I_prev, obs_range, axis=0)
        )
        numpyro.deterministic("prev", jnp.mean(rho_vec) * jnp.take(prev, obs_range, axis=0))

        # Compute growth rate assuming I_{t+1} = I_{t} \exp(r_{t})
        numpyro.deterministic(
            "r",
            jnp.diff(
                jnp.log(jnp.take(I_prev, obs_range, axis=0)),
                prepend=jnp.nan,
                axis=0,
            ),
        )

        # Compute expected cases
        numpyro.deterministic(
            "total_smooth_prev",
            jnp.mean(rho_vec) * jnp.take(I_prev, obs_range),
        )
        EC = numpyro.deterministic("EC", jnp.take(I_prev, obs_range) * rho_vec)

        # Evaluate case likelihood
        CaseLik.model(cases, EC, pred=pred)

        if forecast_L > 0:
            I_forecast = numpyro.deterministic(
                "I_smooth_forecast",
                jnp.mean(rho_vec) * I_prev[(seed_L + T) :],
            )
            numpyro.deterministic(
                "r_forecast",
                jnp.diff(jnp.log(I_forecast), prepend=jnp.nan, axis=0),
            )

    return _model


class SingleRenewalModel(ModelSpec):
    def __init__(
        self,
        g,
        delays,
        seed_L: int,
        forecast_L: int,
        inf_period=None,
        k: Optional[int] = None,
        CLik=None,
        basis_fn: Optional[BasisFunction] = None,
    ):
        self.g_rev = jnp.flip(g, axis=-1)
        self.delays = delays
        self.inf_period = (
            to_survivor_function(inf_period)
            if inf_period is not None
            else jnp.ones(self.g_rev.shape[0])
        )
        self.seed_L = seed_L
        self.forecast_L = forecast_L

        # Making basis expansion for Rt
        self.k = k if k else 10
        self.basis_fn = (
            basis_fn if basis_fn else Spline(s=None, order=4, k=self.k)
        )

        # Defining model likelihoods
        self.CLik = CLik
        self.make_model()

    def make_model(self):
        self.model_fn = _single_renewal_factory(
            self.g_rev,
            self.delays,
            self.inf_period,
            self.seed_L,
            self.forecast_L,
            self.CLik,
        )

    def augment_data(self, data):
        # Add feature matrix for parameterization of R
        data["X"] = self.basis_fn.make_features(data, T=data["cases"].shape[0])
