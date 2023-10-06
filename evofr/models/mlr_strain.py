from typing import Optional
import numpy as np
from evofr.data.data_spec import DataSpec
import jax.numpy as jnp
from jax.nn import softmax, logsumexp

import numpyro
import numpyro.distributions as dist

from .model_spec import ModelSpec


# def mut_dist_ll(pred_freq, next_init_freq, distances, mu, alpha, t, deltaT):
#     # Generate probability each sequence in next gen belongs to projected clade
#
#     # Get individual weights
#     _, S_next = distances.shape
#     with numpyro.plate(f"future_pop_{t}", S_next):
#         sample_identity = numpyro.sample(
#             f"sample_identity_{t}", dist.Dirichlet(pred_freq * alpha / S_next)
#         )
#
#     # Compute mixture probability
#     log_prob_dist = dist.Poisson(mu * deltaT).log_prob(distances)
#     log_prob = (
#         log_prob_dist + jnp.log(sample_identity.T) + jnp.log(next_init_freq)
#     )
#     numpyro.factor(f"distance_{t}", logsumexp(log_prob, axis=0).sum())
#
#     # Re-think
#     # Each sample gets own dirichlet showing similarity to previous generation
#     # Samples which make up more of the population should matter for this
#     # Is there a probabilitisticly grounded way for doing this
#     return None


def mut_dist_ll(pred_freq, next_init_freq, distances, mu, t, deltaT):
    # Compute mixture probability
    log_prob_dist = dist.Poisson(mu * deltaT).log_prob(distances)
    log_prob = (
        log_prob_dist + jnp.log(pred_freq)[:, None] + jnp.log(next_init_freq)
    )
    numpyro.factor(f"distance_{t}", logsumexp(log_prob, axis=0).sum())
    return None


def mut_ancestor_ll(pred_freq, distances, ancestors, mu, t, deltaT):
    # We only need distances between ancestors
    S_now, S_next = distances.shape
    ancestor_distances = (ancestors, np.arange(S_next))

    # Compute distances between ancestors
    numpyro.sample(
        f"ancestor_distances_{t}",
        dist.Poisson(mu * deltaT),
        obs=distances[ancestor_distances],
    )

    # Count unique ancestors and compare to prob of descendence
    ancestor_counts = (ancestors == np.arange(S_now)[:, None]).sum(axis=1)
    numpyro.sample(
        f"ancestor_counts_{t}",
        dist.Multinomial(ancestor_counts.sum(), pred_freq),
        obs=ancestor_counts,
    )
    return None


def strain_distance_numpyro(
    init_freq,
    distances,
    predictors,
    counts,
    ancestors,
    deltaT=1,
    pred=False,
    mu=None,
):
    """Fit a strain distance model using MLR type model
    to project frequencies and compare distances between populations.


    Parameters
    ----------
    init_freq:
        List of length G containing initial frequencies for each generation.
        Each element of list should have shape (S_{g}, ) and sum to one.

    distances:
        List of distances between samples from a generation.
        Each element of list should have shape (S_{g}, S_{g+1}).

    predictors:
        List of predictor matrices for the members of each generation.
        Each element of list should have shape (S_{g}, P).

    counts:
        List of length G containing counts of strains for each generation.
        Each element of list should have shape (S_{g},).

    ancestors:
        List of length G containing ancestors of samples of a generation.
        Each element of the list should have shape (S_{g+1}, ) and
        should contain values between 0 and S_{g}.
    """
    # We'll have a list of distance matrices between time points
    G = len(predictors)

    # We'll also have a list of the initial frequencies in each generation
    # T + 1 = len(init_freq) and [(S_{1},), ..., (S_{T+1})]

    # Get coefficients for each predictors
    P = predictors[0].shape[-1]  # [(S_{1}, P), ..., (S_{T+1}, P)]
    with numpyro.plate("predictors", P):
        coef = numpyro.sample("coefficients", dist.Normal(0.0, 5.0))

    # Get mutation rate: (Expected mutations per unit time)
    if mu is None:
        mu = numpyro.sample("mu", dist.Exponential(0.5))

    # If counts are present, get analytic posterior for frequencies
    if init_freq is None and counts is not None:
        init_freq = []
        for g in range(G):
            init_freq.append(
                numpyro.sample(
                    f"init_freq_{g}", dist.Dirichlet(1.0 + counts[g])
                )
            )

    # Pivot value for fitness
    for g in range(G - 1):
        # Compute fitnesses and project
        fitness = numpyro.deterministic(
            f"fitness_{g}", jnp.dot(predictors[g], coef)
        )  # (S_now, )
        logit_proj_freq = fitness * deltaT + jnp.log(init_freq[g])
        proj_freq = numpyro.deterministic(
            f"proj_freq_{g}", softmax(logit_proj_freq)
        )

        # Get distances and compute likelihood
        if ancestors is None:
            mut_dist_ll(
                proj_freq, init_freq[g + 1], distances[g], mu, g, deltaT
            )
        else:
            mut_ancestor_ll(
                proj_freq, distances[g], ancestors[g], mu, g, deltaT
            )

    if pred:
        # Compute fitnesses and project
        fitness = numpyro.deterministic(
            f"fitness_{G-1}", jnp.dot(predictors[-1], coef)
        )  # (S_now, )
        logit_proj_freq = fitness * deltaT + jnp.log(init_freq[-1])
        proj_freq = numpyro.deterministic(
            f"proj_freq_{G-1}", softmax(logit_proj_freq)
        )


class StrainDistanceModel(ModelSpec):
    def __init__(self, mu=None) -> None:
        """Construct ModelSpec for StrainDistanceModel

        Parameters
        ----------
        ....

        Returns
        -------
        StrainDistanceModel
        """
        self.model_fn = strain_distance_numpyro
        self.mu = mu

    def augment_data(self, data: dict) -> None:
        data["mu"] = self.mu


class StrainDistanceData(DataSpec):
    def __init__(
        self, init_freq, predictors, distances, ancestors=None, counts=None
    ) -> None:
        self.init_freq = init_freq
        self.predictors = predictors
        self.distances = distances
        self.ancestors = ancestors
        self.counts = counts

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()

        data["init_freq"] = self.init_freq
        data["predictors"] = self.predictors
        data["distances"] = self.distances
        data["counts"] = self.counts
        data["ancestors"] = self.ancestors
        return data


####
# TODO: StrainDistanceData
# - Takes in list of strains and timepoints
#   - Also list of prectors by strain info
# Will also need mapping from index to strain name :-D.
# - Also need distances

# TODO: Decide whether we take representative strains or not
# TODO: Add prior rate for mu_prior
