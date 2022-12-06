from abc import ABC, abstractmethod
from typing import List, Optional
from jax._src.nn.functions import softmax
import numpy as np
import pandas as pd

from evofr.data.data_spec import DataSpec
from evofr.data.data_helpers import prep_dates, prep_sequence_counts
from evofr.models.renewal_model.model_options import MultinomialSeq
from .multinomial_logistic_regression import MultinomialLogisticRegression
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from .model_spec import ModelSpec

from functools import partial


class DeltaPriorModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def model(self):
        pass


class DeltaNormalPrior(DeltaPriorModel):
    def __init__(
        self, loc: Optional[float] = None, scale: Optional[float] = None
    ):
        self.loc = loc
        self.scale = scale

    def model(self, N_variants):
        delta_scale = (
            numpyro.sample("delta_scale", dist.HalfNormal(0.1))
            if self.loc is None
            else self.loc
        )
        delta_loc = (
            numpyro.sample("delta_loc", dist.Normal(0.0, 0.1))
            if self.scale is None
            else self.scale
        )
        raw_delta = numpyro.sample(
            "raw_delta",
            dist.Normal(delta_loc, delta_scale),
            sample_shape=(N_variants - 1,),
        )
        return raw_delta


class DeltaRegressionPrior(DeltaPriorModel):
    def __init__(self, features):
        """Construct a regression-based prior for relative fitness innovations.

        Parameters
        ----------
        features:
            Matrix of shape (N_variants, N_features)
            to predict relative fitness.

        Returns
        -------
        DeltaRegressionPrior
        """
        # Creature and process features (N_variants-1, n_features)
        self.features = features[:-1, :]
        self.N_features = self.features.shape[-1]
        self.is_missing = np.isnan(self.features).any(axis=1)
        self.n_missing = self.is_missing.sum()

    def model(self, N_variants):
        # Generate coefficients for features
        theta = numpyro.sample(
            "theta", dist.Normal(), sample_shape=(self.N_features,)
        )

        # Generate relative fitness where features present
        delta_loc_present = jnp.dot(self.features[~self.is_missing, :], theta)
        # Generate relative fitness where features missing
        delta_loc_missing = numpyro.sample(
            "delta_loc_missing",
            dist.Normal(0.0, 3.0),
            sample_shape=(self.n_missing,),
        )

        # Combine delta_loc for missing and present predictors
        delta_loc = jnp.empty(N_variants-1)
        delta_loc = delta_loc.at[~self.is_missing].set(delta_loc_present)
        delta_loc = delta_loc.at[self.is_missing].set(delta_loc_missing)
        numpyro.deterministic("delta_loc", delta_loc)

        delta_scale = numpyro.sample("delta_scale", dist.HalfNormal(0.1))

        raw_delta = numpyro.sample(
            "raw_delta",
            dist.Normal(delta_loc, delta_scale)
        )
        return raw_delta

    def predict(self, features, samples):
        theta = samples["theta"]
        delta_scale = samples["delta_scale"]

        # Compute mean of delta
        delta_loc = jnp.einsum("vf, sf -> sv", features, theta)
        prior_delta = np.random.normal(delta_loc, delta_scale)
        return delta_loc, prior_delta


def MLR_innovation_model(
    seq_counts,
    N,
    X,
    innovation_matrix,
    delta_prior,
    tau=None,
    pred=False,
):
    T, N_variants = seq_counts.shape
    _, N_features = X.shape

    # Sampling parameters for growth advantages
    # Sampling intercepts
    raw_alpha = numpyro.sample(
        "raw_alpha",
        dist.Normal(0.0, 3.0),
        sample_shape=(N_variants - 1,),
    )
    raw_alpha = jnp.append(raw_alpha, jnp.zeros(1))

    # Sampling innovations from prior model
    raw_delta = delta_prior.model(N_variants)

    # We need last variant to have 0 growth advantage
    # so set its delta to negative of parent contributions
    pivot_delta = jnp.dot(raw_delta, innovation_matrix[-1, :-1])
    delta = jnp.append(raw_delta, -pivot_delta)
    numpyro.deterministic("delta", delta)

    # Innovations to beta for growth advantages
    raw_beta = jnp.dot(innovation_matrix, delta)

    beta = numpyro.deterministic(
        "beta", jnp.column_stack((raw_alpha, raw_beta)).T
    )

    logits = jnp.dot(X, beta)  # Logit frequencies by variant

    # Compute frequency
    # _freq = jnp.exp(logits)
    # _freq = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    numpyro.deterministic("freq", softmax(logits, axis=-1))

    # Evaluate likelihood of sequence counts given delays
    # SeqLik.model(seq_counts, N, freq, pred)  # Evaluate likelihood

    obs = None if pred else np.nan_to_num(seq_counts)
    numpyro.sample(
        "seq_counts",
        dist.Multinomial(total_count=np.nan_to_num(N), logits=logits),
        obs=obs,
    )

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic("ga", jnp.exp(raw_beta * tau)[:-1])
        numpyro.deterministic("ga_delta", jnp.exp(delta * tau))


class InnovationMLR(ModelSpec):
    def __init__(
        self, tau: float, delta_prior: Optional[DeltaPriorModel] = None
    ) -> None:
        """Construct ModelSpec for MLR allowing derived variants
        to share past fitness innovations.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        prior_model:
            Prior model for variant growtha advantage innovations.

        Returns
        -------
        InnovationMLR
        """
        self.tau = tau  # Fixed generation time
        self.delta_prior = (
            DeltaNormalPrior() if delta_prior is None else delta_prior
        )
        self.model_fn = partial(
            MLR_innovation_model, tau=self.tau, delta_prior=self.delta_prior
        )

    def augment_data(self, data: dict) -> None:
        T = len(data["N"])
        data["X"] = MultinomialLogisticRegression.make_ols_feature(
            0, T
        )  # Use intercept and time as predictors


def prep_clade_list(
    raw_variant_parents: pd.DataFrame,
    var_names: Optional[List] = None,
):
    """Process 'raw_variant_parents' data to nd.array showing
    which innovations present by variant.

    Parameters
    ----------
    raw_variant_parents:
        a dataframe containing variant name and parent variant

    var_names:
        optional list of variants

    pivot:
        optional name of variant to place last.
        Defaults to "other" if present otherwise.
        This will usually used as a reference or pivot strain.

    Returns
    -------
    innovation_matrix:
        binary matrix showing whether A[i,j] = i has innovation from j.
    """

    # First check:
    # SHould be able to reduce to a faster solve maybe?
    var_names = (
        list(raw_variant_parents.variant.values)
        if var_names is None
        else var_names
    )
    innovation_matrix = np.zeros((len(var_names), len(var_names))).astype(int)
    parent_map = {
        row["variant"]: row["parent"]
        for _, row in raw_variant_parents.iterrows()
    }
    for _, row in raw_variant_parents.iterrows():
        var = row["variant"]
        var_index = var_names.index(var)

        # Find ancestors and add to row
        par = parent_map[var]
        is_par_variant = par in var_names
        while is_par_variant:
            par_index = var_names.index(par)  # Find index
            innovation_matrix[var_index, par_index] = 1  # Add innovation
            par = parent_map[par]  # New parent is parent of previous
            is_par_variant = par in var_names  # Check if should continue
        innovation_matrix[var_index, var_index] = 1
    return innovation_matrix.astype(bool), parent_map


class InnovationSequenceCounts(DataSpec):
    def __init__(
        self,
        raw_seq: pd.DataFrame,
        raw_variant_parents: pd.DataFrame,
        date_to_index: Optional[dict] = None,
        var_names: Optional[List] = None,
        pivot: Optional[str] = None,
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

        Returns
        -------
        InnovationSequenceCounts
        """

        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_seq["date"])
        self.date_to_index = date_to_index

        # Turn dataframe to counts of each variant sequenced each day
        self.var_names, self.seq_counts = prep_sequence_counts(
            raw_seq, self.date_to_index, var_names, pivot=pivot
        )
        self.pivot = self.var_names[-1]

        # Create innovation_matrix
        self.innovation_matrix, self.parent_map = prep_clade_list(
            raw_variant_parents, self.var_names
        )

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()

        # Export `seq_counts`, sequences each day, and variant names
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=-1)
        data["innovation_matrix"] = self.innovation_matrix
        return data
