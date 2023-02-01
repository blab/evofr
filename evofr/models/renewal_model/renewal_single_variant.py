from typing import Optional
from functools import partial
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


def _single_renewal_model(
    g_rev,
    delays,
    inf_period,
    seed_L,
    forecast_L,
    cases,
    X,
    day_of_week_effect=True,
    CaseLik=None,
    pred=False,
):
    if CaseLik is None:
        CaseLik = NegBinomCases()

    T, k = X.shape
    obs_range = jnp.arange(seed_L, seed_L + T, 1)

    # Effective Reproduction number likelihood
    gam = numpyro.sample("gam", dist.HalfNormal(1.0))
    beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 1.0))
    with numpyro.plate("N_steps_base", k - 1):
        beta_rw_step = numpyro.sample("beta_rw_step", dist.Laplace()) * gam
        beta_rw = numpyro.deterministic("beta_rw", jnp.cumsum(beta_rw_step))

    # Combine increments and starting position
    beta_rw = jnp.append(jnp.zeros(1), beta_rw)
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
    logI0 = numpyro.sample("logI0", dist.Normal()) * 3.0 + 1.0
    I0 = jnp.exp(logI0)
    intros = jnp.zeros((T + seed_L + forecast_L,))
    intros = intros.at[np.arange(seed_L)].set(I0 * jnp.ones(seed_L))

    # Generate day-of-week reporting fraction

    if day_of_week_effect:
        with numpyro.plate("rho_parms", 6):
            rho_logits = numpyro.sample("rho_logits", dist.Normal()) * 5.0
        _rho = jnp.exp(jnp.append(rho_logits, 0.0))
        rho = numpyro.deterministic("rho", _rho / _rho.sum())
    else:
        rho = jnp.ones(7)
    rho_vec = reporting_to_vec(rho, T)

    I_prev, prev = forward_simulate_I_and_prev(
        intros, R, g_rev, delays, inf_period, seed_L
    )
    I_prev = jnp.clip(I_prev, a_min=1e-12, a_max=1e25)

    # Smooth trajectory for plotting
    numpyro.deterministic(
        "I_smooth", jnp.mean(rho_vec) * jnp.take(I_prev, obs_range, axis=0)
    )
    numpyro.deterministic(
        "prev", jnp.mean(rho_vec) * jnp.take(prev, obs_range, axis=0)
    )

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
    return None


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
        day_of_week_effect: bool = True,
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
        self.day_of_week_effect = day_of_week_effect
        self.CLik = CLik
        self.make_model()

    def make_model(self):
        self.model_fn = partial(
            _single_renewal_model,
            g_rev=self.g_rev,
            delays=self.delays,
            inf_period=self.inf_period,
            seed_L=self.seed_L,
            forecast_L=self.forecast_L,
            day_of_week_effect=self.day_of_week_effect,
            CaseLik=self.CLik,
        )

    def augment_data(self, data):
        # Add feature matrix for parameterization of R
        data["X"] = self.basis_fn.make_features(data, T=data["cases"].shape[0])
