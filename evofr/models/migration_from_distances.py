from typing import Optional
import numpy as np
from evofr.data.data_spec import DataSpec
import jax.numpy as jnp
from jax.nn import softmax, logsumexp

import numpyro
import numpyro.distributions as dist

from .model_spec import ModelSpec


def mut_dist_mig_ll(
    distances, locations, locations_next, n_locations, mu, t, deltaT, alpha
):

    # Get individual weights
    # S_now, S_next = distances.shape

    # with numpyro.plate(f"mig_step_{t}", S_next):
    #    sample_identity = numpyro.sample(
    #        f"sample_identity_{t+1}",
    #        dist.Dirichlet(np.ones(S_now) / S_now * alpha),
    #    )  # (S_next, S_now)

    # Compute mixture probability
    # Probability of distance assuming descendent
    log_prob_dist = dist.Poisson(mu * deltaT).log_prob(distances)

    # Model probabiltiy of descendent with softmax on distnace
    sample_identity = softmax(-distances / alpha, axis=0)

    # Weight by probability of parentage "sample_identity"
    log_prob = log_prob_dist + jnp.log(sample_identity)
    numpyro.factor(f"distance_{t}", logsumexp(log_prob, axis=0).sum())

    # Use sample locations to generate rates between regions
    # Map samples to locations
    location_mat = (
        np.arange(n_locations) == locations[:, None]
    )  # (S_now, n_locations)
    location_mat_next = (
        np.arange(n_locations) == locations_next[:, None]
    )  # (S_next, n_locations)
    num_in_location_next = location_mat_next.sum(axis=0)

    # Probability of sample originating from region
    p_sample_loc = numpyro.deterministic(
        f"parent_loc_{t+1}",
        jnp.einsum("pl, pc -> cl", location_mat, sample_identity),
    )  # (S_next, n_locations)
    # Average probability of moving between locations
    p_loc_to_loc = numpyro.deterministic(
        f"mig_{t+1}",
        jnp.einsum("cm, cl -> lm", location_mat_next, p_sample_loc)
        / num_in_location_next,
    )  # (n_locations, n_locations)

    # TODO: Smooth between time points. Do I put time varying latent rate on this?

    return sample_identity, p_sample_loc, p_loc_to_loc


def mut_dist_ll(pred_freq, next_init_freq, distances, mu, t, deltaT):
    # Compute mixture probability
    log_prob_dist = dist.Poisson(mu * deltaT).log_prob(distances)
    log_prob = (
        log_prob_dist + jnp.log(pred_freq)[:, None] + jnp.log(next_init_freq)
    )
    numpyro.factor(f"distance_{t}", logsumexp(log_prob, axis=0).sum())
    return None


def migration_distance_numpyro(
    distances,
    locations,
    n_locations,
    deltaT=1,
    pred=False,
    mu=None,
    alpha=None,
):
    """Fit a strain distance model using MLR type model
    to project frequencies and compare distances between populations.


    Parameters
    ----------
    distances:
        List of distances between samples from a generation.
        Each element of list should have shape (S_{g}, S_{g+1}).

    locations:
        List of locations of samples from a generation.
        Each element of list should have shape (S_{g}, ).
    """
    # We'll have a list of distance matrices between time points
    G = len(locations)

    # Get mutation rate per unit time
    if mu is None:
        mu = numpyro.sample("mu", dist.Exponential(1.0))

    # Get temperature for distnace
    if alpha is None:
        alpha = numpyro.sample("alpha", dist.Exponential(1.0))

    for g in range(G - 1):
        # Get distances and compute probability of belonging
        mut_dist_mig_ll(
            distances[g],
            locations[g],
            locations[g + 1],
            n_locations,
            mu,
            g,
            deltaT,
            alpha,
        )

    if pred:
        # Simulate original multinomial for each and then do same operations on counts
        print("Predicting")


class DistanceMigrationModel(ModelSpec):
    def __init__(self) -> None:
        """Construct ModelSpec for DistanceMigrationModel

        Parameters
        ----------
        ....

        Returns
        -------
        DistanceMigrationModel
        """
        self.model_fn = migration_distance_numpyro

    def augment_data(self, data: dict) -> None:
        pass


class DistanceMigrationData(DataSpec):
    def __init__(
        self, distances, locations, predictors: Optional[list] = None
    ) -> None:
        self.distances = distances
        self.locations = locations
        self.predictors = predictors

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()

        data["distances"] = self.distances
        data["locations"] = self.locations
        data["n_locations"] = 3  # Really needs actual total number
        if self.predictors is not None:
            data["predictors"] = self.predictors
        return data
