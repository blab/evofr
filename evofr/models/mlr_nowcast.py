from abc import ABC, abstractmethod
from typing import Callable, List, Optional
import numpy as np
import pandas as pd

from evofr.data.data_spec import DataSpec
from evofr.data.data_helpers import prep_dates, format_var_names
from evofr.models.renewal_model.model_options import MultinomialSeq
from .multinomial_logistic_regression import MultinomialLogisticRegression
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist

from .model_spec import ModelSpec

from functools import partial
from .renewal_model import Spline


class HazardModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def model(self, N_variants, D):
        pass


class LinearHazard(HazardModel):
    def __init__(self):
        pass

    def model(self, N_variants, D):
        # Assume hazard is a linear function
        # This is pooled so that new variants are closer to mean
        a_mean = numpyro.sample("a_mean", dist.Normal(0.0, 1.0)) * 5.0
        b_mean = numpyro.sample("b_mean", dist.Normal(0.0, 1.0)) * 5.0
        with numpyro.plate("variant", N_variants):
            a = numpyro.sample("a", dist.Normal(0.0, 1.0)) * 5.0 + a_mean
            b = numpyro.sample("b", dist.Normal(0.0, 1.0)) * 5.0 + b_mean
        logit_h = a[:, None] * jnp.arange(D - 1) + b[:, None]
        _h = jnp.exp(logit_h) / (1 + jnp.exp(logit_h))
        return _h


class BetaHazard(HazardModel):
    def __init__(self):
        pass

    def model(self, N_variants, D):
        _h_conc = numpyro.sample("_h_conc", dist.InverseGamma(2.0, 2.0))
        with numpyro.plate("delay", D - 1):
            _h_mean = numpyro.sample("_h_mean", dist.Beta(0.1, 0.1))
            with numpyro.plate("variant", N_variants):
                _h = numpyro.sample(
                    "_h",
                    dist.BetaProportion(mean=_h_mean, concentration=_h_conc),
                )
        return _h


class LogitRWHazard(HazardModel):
    def __init__(self):
        pass

    def model(self, N_variants, D):
        h_scale = numpyro.sample("h_scale", dist.HalfNormal(1.0))
        with numpyro.plate("variant", N_variants):
            logit_h = numpyro.sample(
                "logit_h",
                dist.GaussianRandomWalk(scale=h_scale, num_steps=D - 1),
            )
        _h = jnp.exp(logit_h) / (1 + jnp.exp(logit_h))
        return _h


class LogitSplineHazard(HazardModel):
    def __init__(
        self,
        k: Optional[int] = None,
        order: Optional[int] = None,
        pool_scale: Optional[float] = None,
    ):
        self.k = 4 if k is None else k
        self.order = 4 if order is None else order
        self.pool_scale = 0.2 if pool_scale is None else pool_scale

    def model(self, N_variants, D):
        U = Spline(k=self.k, order=self.order).make_features(T=D - 1)
        _, k = U.shape

        # gammma_scale = numpyro.sample("gamma_scale", dist.HalfNormal(0.1))
        with numpyro.plate("basis_function", k):
            gamma_loc = (
                numpyro.sample("gamma_loc", dist.Normal(0.0, 1.0)) * 5.0
            )
            with numpyro.plate("variant", N_variants):
                gamma = numpyro.sample(
                    "gamma", dist.Normal(gamma_loc, self.pool_scale)
                )

        logit_h = numpyro.deterministic("logit_h", jnp.dot(U, gamma.T))
        _h = jnp.exp(logit_h) / (1 + jnp.exp(logit_h))
        return _h.T


def discrete_hazard_to_pmf_cdf(h):
    V, _ = h.shape
    # survival = jnp.exp(jnp.cumsum(jnp.log(1 - h), axis=1))
    survival = jnp.cumprod(1 - h, axis=1)
    survival = jnp.append(survival, jnp.zeros((V, 1)), 1)
    pmf = jnp.diff(1 - survival, axis=1, prepend=0.0)
    return pmf, 1 - survival


def estimate_delay(seq_count_delays, hazard_model):
    _, N_variants, D = seq_count_delays.shape

    # Assume delay distribution is constant over time
    empirical_delays = np.nan_to_num(seq_count_delays.sum(axis=0))
    total_of_variant = np.nan_to_num(empirical_delays.sum(axis=-1))

    # Generate discrete based on specified hazard model
    h = numpyro.deterministic("h", hazard_model(N_variants, D))

    # Get delay distribution
    p_delay, cdf_delay = discrete_hazard_to_pmf_cdf(h)
    p_delay = numpyro.deterministic(
        "p_delay", jnp.clip(p_delay, 1e-12, 1 - 1e-12).T
    )
    cdf_delay = numpyro.deterministic(
        "cdf_delay", jnp.clip(cdf_delay, 1e-12, 1).T
    )

    # Evaluate likelihood of delays
    numpyro.sample(
        "delays",
        dist.Multinomial(total_count=total_of_variant, probs=p_delay.T),
        obs=empirical_delays,
    )

    return p_delay, cdf_delay


def MLR_nowcast_model(
    seq_counts,
    seq_counts_delay,
    N,
    X,
    hazard_model,
    SeqLik,
    tau=None,
    pred=False,
):
    T, N_variants, D = seq_counts_delay.shape
    _, N_features = X.shape

    # Get time since present used to estimate fraction
    # of variant samples observed thus far.
    time_before_now = np.minimum(T - np.arange(T), D) - 1

    # Sampling parameters for growth advantages
    raw_beta = (
        numpyro.sample(
            "raw_beta",
            dist.Normal(0.0, 1.0),
            sample_shape=(N_features, N_variants - 1),
        )
        * jnp.array([10.0, 1.0])[:, None]
    )

    beta = numpyro.deterministic(
        "beta",
        jnp.column_stack(
            (raw_beta, jnp.zeros(N_features))
        ),  # All parameters are relative to last column / variant
    )

    logits = jnp.dot(X, beta)  # Logit frequencies by variant

    # Compute frequency
    _freq = jnp.exp(logits)
    freq = numpyro.deterministic(
        "freq", jnp.clip(_freq / _freq.sum(axis=-1)[:, None], 1e-12, 1 - 1e-12)
    )

    # Estimate delay distribution by variant
    _, cdf_delay = estimate_delay(seq_counts_delay, hazard_model.model)

    # Get adjusted frequency
    frac_delay = numpyro.deterministic(
        "frac_delay", jnp.take(cdf_delay, time_before_now, axis=0)
    )  # (T, V)

    # Correct recent frequencies by probability that
    # collected sequences have been submitted by observation date.
    _nowcast_freq = frac_delay * freq
    nowcast_freq = _nowcast_freq / _nowcast_freq.sum(axis=-1)[:, None]
    nowcast_freq = jnp.clip(nowcast_freq, 1e-16, 1 - 1e-16)

    # Evaluate likelihood of sequence counts given delays
    SeqLik.model(seq_counts, N, nowcast_freq, pred)  # Evaluate likelihood

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(beta[-1, :-1] * tau)
        )  # Last row corresponds to linear predictor / growth advantage


class MLRNowcast(ModelSpec):
    def __init__(
        self,
        tau: float,
        hazard_model: Optional[HazardModel] = None,
        SeqLik=None,
    ) -> None:
        """Construct ModelSpec for MLR accounting for submission delay
        differences betwen variants.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        hazard_model:
            Numpyro model describing the hazard function for submission delay.

        SeqLik:
            Optional sequence likelihood option: MultinomialSeq or
            DirMultinomialSeq. Defaults to MultinomialSeq.

        Returns
        -------
        MLRNowcast
        """
        self.tau = tau  # Fixed generation time
        self.hazard_model = (
            LinearHazard if hazard_model is None else hazard_model
        )
        self.SeqLik = MultinomialSeq() if SeqLik is None else SeqLik
        self.model_fn = partial(
            MLR_nowcast_model,
            hazard_model=self.hazard_model,
            SeqLik=self.SeqLik,
        )

    def augment_data(self, data: dict) -> None:
        T = len(data["N"])
        data["tau"] = self.tau
        data["X"] = MultinomialLogisticRegression.make_ols_feature(
            0, T
        )  # Use intercept and time as predictors


def prep_sequence_counts_delay(
    raw_seqs: pd.DataFrame,
    date_to_index: Optional[dict] = None,
    var_names: Optional[List] = None,
    max_delay: Optional[int] = None,
    pivot: Optional[str] = None
):
    """Process 'raw_seq' data to nd.array including unobserved dates.

    Parameters
    ----------
    raw_seq:
        a dataframe containing sequence counts with
        columns 'sequences' and 'date'.


    date_to_index:
        optional dictionary for mapping calender dates to nd.array indices.

    var_names:
        optional list of variant to count observations.

    pivot:
        optional name of variant to place last.
        Defaults to "other" if present otherwise.
        This will usually used as a reference or pivot strain.

    Returns
    -------
    var_names:
        list of variants counted

    C:
        nd.array containing number of sequences of each variant on each date.
    """

    if var_names is None:
        raw_var_names = list(pd.unique(raw_seqs.variant))
        raw_var_names.sort()
        var_names = format_var_names(raw_var_names, pivot)

    raw_seqs["date"] = pd.to_datetime(raw_seqs["date"])
    if date_to_index is None:
        _, date_to_index = prep_dates(raw_seqs["date"])

    # Build matrix
    T = len(date_to_index)
    V = len(var_names)
    D = (
        min(raw_seqs["delay"].max(), max_delay)
        if max_delay is not None
        else raw_seqs["delay"].max()
    )
    C = np.full((T, V, D), np.nan)

    # Loop over rows in dataframe to fill matrix
    for i, s in enumerate(var_names):
        for _, row in raw_seqs[raw_seqs.variant == s].iterrows():
            delay = min(row.delay, D)
            C[date_to_index[row.date], i, delay - 1] = row.sequences

    # Loop over matrix rows to correct for zero counts
    for d in range(T):
        if not np.isnan(C[d, :, :]).all():
            # If not all days samples are nan, replace nan with zeros
            np.nan_to_num(C[d, :, :], copy=False)
    return var_names, C


class DelaySequenceCounts(DataSpec):
    def __init__(
        self,
        raw_seq: pd.DataFrame,
        date_to_index: Optional[dict] = None,
        var_names: Optional[List] = None,
        max_delay: Optional[int] = None,
        pivot: Optional[str] = None
    ):
        """Construct a data specification for handling variant frequencies.

        Parameters
        ----------
        raw_seq:
            a dataframe containing sequence counts with columns 'sequences',
            'variant', and date'.

        date_to_index:
            optional dictionary for mapping calender dates to nd.array indices.

        var_names:
            optional list containing names of variants to be present

        max_delay:
            optional integer showing the maximum allowed submission delay.

        pivot:
            optional name of variant to place last.
            Defaults to "other" if present otherwise.
            This will usually used as a reference or pivot strain.

        Returns
        -------
        DelaySequenceCounts
        """

        # Get mapping from date to index
        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_seq["date"])
        self.date_to_index = date_to_index

        # Turn dataframe to counts of each variant sequenced each day
        self.pivot = pivot
        self.var_names, self.seq_counts_delay = prep_sequence_counts_delay(
            raw_seq, self.date_to_index, var_names, max_delay, self.pivot
        )
        self.seq_counts = self.seq_counts_delay.sum(axis=-1)

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()

        # Export `seq_counts`, sequences each day, and variant names
        data["seq_counts_delay"] = self.seq_counts_delay
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=-1)
        return data
