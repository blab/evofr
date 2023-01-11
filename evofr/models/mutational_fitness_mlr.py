from typing import List, Optional
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


def mutational_fitness_model(
    seq_counts,
    N,
    X,
    mutation_presence,
    SeqLik,
    tau=None,
    pred=False,
):
    T, N_variants = seq_counts.shape
    _, N_features = X.shape
    _, N_mutations = mutation_presence.shape

    # Sampling parameters for growth advantages
    # Sampling intercepts
    raw_alpha = (
        numpyro.sample(
            "raw_alpha",
            dist.Normal(0.0, 1.0),
            sample_shape=(1, N_variants - 1),
        )
        * 10.0
    )
    raw_alpha = jnp.append(raw_alpha, 0.0)

    # Sampling mutation innovations
    raw_delta = numpyro.sample(
        "raw_delta", dist.Normal(0.0, 1.0), sample_shape=(N_mutations,)
    )
    # Innovations to beta
    raw_beta = jnp.dot(mutation_presence, raw_delta)
    raw_beta = raw_beta - raw_beta[-1]

    # How do we ensure identifiablity?
    # We need to make sure everything is relative to pivot
    # But we won't be able to estimate mutational fitness of things which ...
    # We can have effect by lineage for re-occuring mutations :-D.

    beta = numpyro.deterministic(
        "beta", jnp.row_stack((raw_alpha, raw_beta))
    )  # All parameters are relative to last column / variant

    logits = jnp.dot(X, beta)  # Logit frequencies by variant

    # Compute frequency
    _freq = jnp.exp(logits)
    freq = numpyro.deterministic(
        "freq", jnp.clip(_freq / _freq.sum(axis=-1)[:, None], 1e-12, 1 - 1e-12)
    )

    # Evaluate likelihood of sequence counts given delays
    SeqLik.model(seq_counts, N, freq, pred)  # Evaluate likelihood

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic("ga", jnp.exp(raw_beta * tau)[:-1])


class MutationalFitnessMLR(ModelSpec):
    def __init__(
        self,
        tau: float,
        SeqLik=None,
    ) -> None:
        """Construct ModelSpec for MLR allowing derived variants
        to have mutation derived fitnesses.

        Parameters
        ----------
        tau:
            Assumed generation time for conversion to relative R.

        SeqLik:
            Optional sequence likelihood option: MultinomialSeq or
            DirMultinomialSeq. Defaults to MultinomialSeq.

        Returns
        -------
        MutationalFitnessMLR
        """
        self.tau = tau  # Fixed generation time
        self.SeqLik = MultinomialSeq() if SeqLik is None else SeqLik
        self.model_fn = partial(
            mutational_fitness_model,
            SeqLik=self.SeqLik,
            tau=self.tau,
        )

    def augment_data(self, data: dict) -> None:
        T = len(data["N"])
        data["X"] = MultinomialLogisticRegression.make_ols_feature(
            0, T
        )  # Use intercept and time as predictors


def prep_mutations(raw_mutations, var_names):
    # Fitler to only variants in var_names
    rw = raw_mutations[raw_mutations.variant.isin(var_names)]

    # Find mutations of interest
    mutations = rw["mutation"].unique()
    num_variants, num_muts = len(var_names), mutations.size
    mut_matrix = np.zeros((num_variants, num_muts))
    for v, variant in enumerate(var_names):
        v_muts = rw[rw.variant == variant]["mutation"].values
        mut_matrix[v, :] = np.isin(mutations, v_muts)

    # Drop rows that are present in everything...
    is_fixed = mut_matrix.sum(axis=0) == num_variants
    return mutations[~is_fixed], mut_matrix[:, ~is_fixed]


class MutationalFitnessSequenceCounts(DataSpec):
    def __init__(
        self,
        raw_seq: pd.DataFrame,
        raw_mutations: pd.DataFrame,
        date_to_index: Optional[dict] = None,
        var_names: Optional[List] = None,
        pivot: Optional[str] = None,
    ):
        """Construct a data specification for handling variant frequencies
        for mutational fitness models.

         Parameters
         ----------
         raw_seq:
             a dataframe containing sequence counts with columns 'sequences',
             'variant', and date'.

         raw_mutations:
             a dataframe containing variants and their mutations with columns
             'variant' and 'mutatation'.

         date_to_index:
             optional dictionary for mapping calender dates to nd.array indices

         var_names:
             optional list containing names of variants to be present

         Returns
         -------
        MutationalFitnessSequenceCounts
        """

        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_seq["date"])
        self.date_to_index = date_to_index

        # Turn dataframe to counts of each variant sequenced each day
        self.var_names, self.seq_counts = prep_sequence_counts(
            raw_seq, self.date_to_index, var_names, pivot=pivot
        )
        self.pivot = self.var_names[-1]

        # Create mutation indicator matrix
        self.mut_names, self.mutation_presence = prep_mutations(
            raw_mutations, self.var_names
        )

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()

        # Export `seq_counts`, sequences each day, and variant names
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=-1)
        data["mutation_presence"] = self.mutation_presence
        return data
