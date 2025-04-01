from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.nn import softmax
from numpyro.distributions.distribution import TransformedDistribution
from numpyro.distributions.transforms import OrderedTransform

import evofr as ef
from evofr.models.relative_fitness_hsgp import HSGaussianProcess

from .model_spec import ModelSpec


class LatentRW:
    def __init__(self):
        pass

    def build_model(self, data):
        self.N_time = data["seq_counts"].shape[0]

    def model_group(self, dim, N_groups):
        N_time = self.N_time
        gam = numpyro.sample(
            "gam",
            dist.TransformedDistribution(
                dist.HalfNormal(), dist.transforms.AffineTransform(0.0, 0.1)
            ),
        )
        # Compute initial values for latent immune in each dimension
        # Note: We fix first component to be ordered for identifiablity
        _phi_0_base = numpyro.sample(
            "_phi_0_base",
            TransformedDistribution(
                dist.Normal(1.0).expand([dim - 1]), OrderedTransform()
            ),
        )  # (dim-1, )
        _phi_0_base = jnp.flip(_phi_0_base)  # First component is highest
        _phi_0_rest = numpyro.sample(
            "_phi_0_rest", dist.Normal().expand([dim - 1, N_groups - 1])
        )

        with numpyro.plate("group", N_groups, dim=-1):
            # Reshaping initial phi values
            _phi_0 = jnp.hstack((_phi_0_base[:, None], _phi_0_rest))
            phi_0 = numpyro.deterministic(
                "phi_0", jnp.vstack((_phi_0, jnp.zeros((1, N_groups))))
            )

            with numpyro.plate("N_dim_m1_phi", dim - 1, dim=-2):
                with numpyro.plate("N_steps_base", N_time - 1, dim=-3):
                    phi_rw_step = numpyro.sample(
                        "phi_rw_step",
                        dist.TransformedDistribution(
                            dist.Normal(0.0, 1.0),
                            dist.transforms.AffineTransform(0.0, gam),
                        ),
                    )
                    phi_rw = jnp.cumsum(phi_rw_step, axis=1)

        # Combine latent factor increments and starting position
        phi_rw = jnp.vstack((jnp.zeros((1, dim - 1, N_groups)), phi_rw))
        phi_rw = jnp.concatenate((phi_rw, jnp.zeros((N_time, 1, N_groups))), axis=1)
        phi = numpyro.deterministic("phi", softmax(phi_0[None, :, :] + phi_rw, axis=1))
        return phi


class LatentSplineRW:
    def __init__(self, basis_fn):
        self.basis_fn = basis_fn

    def build_model(self, data):
        self.X = self.basis_fn.make_features(data)

    def model_group(self, dim, N_groups):
        # Unpacking class properties
        X = self.X

        _, num_knots = X.shape
        gam = numpyro.sample(
            "gam",
            dist.TransformedDistribution(
                dist.HalfNormal(), dist.transforms.AffineTransform(0.0, 0.1)
            ),
        )

        # Compute initial values for latent immune in each dimension
        # Note: We fix first component to be ordered for identifiablity
        _phi_0_base = numpyro.sample(
            "_phi_0_base",
            TransformedDistribution(
                dist.Normal(1.0).expand([dim - 1]), OrderedTransform()
            ),
        )  # (dim-1, )
        _phi_0_base = jnp.flip(_phi_0_base)  # First component is highest
        _phi_0_rest = numpyro.sample(
            "_phi_0_rest", dist.Normal().expand([dim - 1, N_groups - 1])
        )

        with numpyro.plate("group", N_groups, dim=-1):
            # Reshaping initial phi values
            _phi_0 = jnp.hstack((_phi_0_base[:, None], _phi_0_rest))
            phi_0 = numpyro.deterministic(
                "phi_0", jnp.vstack((_phi_0, jnp.zeros((1, N_groups))))
            )

            # Generating splines for latent factors
            with numpyro.plate("N_dim_m1_phi", dim - 1, dim=-2):
                with numpyro.plate("N_steps_base", num_knots - 1, dim=-3):
                    phi_rw_step = numpyro.sample(
                        "phi_rw_step",
                        dist.TransformedDistribution(
                            dist.Normal(0.0, 1.0),
                            dist.transforms.AffineTransform(0.0, gam),
                        ),
                    )
                    phi_rw = jnp.cumsum(phi_rw_step, axis=1)

        # Combine latent factor increments and starting position
        phi_rw = jnp.vstack((jnp.zeros((1, dim - 1, N_groups)), phi_rw))
        phi_rw = jnp.concatenate((phi_rw, jnp.zeros((num_knots, 1, N_groups))), axis=1)
        phi_rw = jnp.einsum("bsg, tb -> tsg", phi_rw, X)
        phi = numpyro.deterministic("phi", softmax(phi_0[None, :, :] + phi_rw, axis=1))
        return phi


class LatentHGSP:
    def __init__(self, hsgp: HSGaussianProcess):
        self.hsgp = hsgp

    def build_model(self, data):
        N_time = data["seq_counts"].shape[0]
        self.features = self.hsgp.make_features(np.arange(N_time))

    def model_group(self, dim, N_groups):
        features, hsgp = self.features, self.hsgp
        N_time, num_basis = features.shape

        # Compute initial coefficients for latent immune in each dimension
        # Note: We fix first component to be ordered for identifiablity
        beta_base = numpyro.sample(
            "beta_base",
            TransformedDistribution(
                dist.Normal(1.0).expand([dim - 1]), OrderedTransform()
            ),
        )  # (dim-1, )
        beta_base = jnp.flip(beta_base)  # First component is highest
        beta_base_rest = numpyro.sample(
            "beta_rest", dist.Normal().expand([dim - 1, N_groups - 1])
        )

        with numpyro.plate("group", N_groups, dim=-1):
            # Reshaping initial beta values
            intercept = jnp.hstack(
                (beta_base[:, None], beta_base_rest)
            )  # (dim-1, N_groups)

            # Generating coefficients for GP basis functions
            with numpyro.plate("N_dim_m1_phi", dim - 1, dim=-2):
                with numpyro.plate("N_bases", num_basis, dim=-3):
                    beta = numpyro.sample(
                        "beta", dist.Normal()
                    )  # (num_basis, dim-1, N_group)

        # Get spectral density
        spd = numpyro.deterministic("sqrt_spd", jnp.sqrt(hsgp.model()))

        # Compute phi and transform to probabilities
        _phi = numpyro.deterministic(
            "_phi",
            jnp.einsum("tj, jsg  -> tsg", features, spd[..., None, None] * beta)
            + intercept[None, :],
        )
        # (N_time, num_basis) * (num_basis, dim-1, N_group) -> (N_time, dim-1, N_group)

        phi = jnp.stack((_phi, jnp.zeros((N_time, 1, N_groups))), axis=1)

        phi = numpyro.deterministic("phi", softmax(phi, axis=1))
        return phi


def relative_fitness_dr_hier_numpyro(
    seq_counts, N, dim, phi_model, tau=None, pred=False, var_names=None
):
    N_time, N_variants, N_groups = seq_counts.shape

    # Sample weights and latent factors
    # Weights are fixed by variant
    with numpyro.plate("N_dim_m1_eta", dim, dim=-1):
        with numpyro.plate("N_variant_weight", N_variants - 1, dim=-2):
            _eta = numpyro.sample("_eta", dist.Uniform(-1.0, 1.0))
            eta = numpyro.deterministic("eta", jnp.vstack((_eta, jnp.zeros((1, dim)))))

    # Latent factors can vary by group
    phi = phi_model.model_group(dim, N_groups)

    # Compute fitness from weights and latent factors
    fitness = jnp.einsum("tsg, sv -> tvg", phi, eta.T)
    numpyro.deterministic("delta", fitness[:, :-1, :])

    # Sample initial frequency
    with numpyro.plate("group_init", N_groups, dim=-1):
        with numpyro.plate("N_variant_init", N_variants - 1, dim=-2):
            _init_logit = numpyro.sample(
                "_init_logit",
                dist.TransformedDistribution(
                    dist.Normal(0.0, 1.0),
                    dist.transforms.AffineTransform(0.0, 8.0),
                ),
            )

    # Sum fitness to get dynamics over time
    init_logit = jnp.vstack((_init_logit, jnp.zeros((1, N_groups))))
    logits = jnp.cumsum(fitness.at[0, :, :].set(0), axis=0) + init_logit

    # Evaluate likelihood
    obs = None if pred else np.swapaxes(np.nan_to_num(seq_counts), 1, 2)
    _seq_counts = numpyro.sample(
        "_seq_counts",
        dist.MultinomialLogits(
            logits=jnp.swapaxes(logits, 1, 2), total_count=np.nan_to_num(N)
        ),
        obs=obs,
    )

    # Re-ordering so groups are last
    seq_counts = numpyro.deterministic("seq_counts", jnp.swapaxes(_seq_counts, 2, 1))

    # Compute frequency
    numpyro.deterministic("freq", softmax(logits, axis=1))

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(fitness[:, :-1, :] * tau)
        )  # Last row corresponds to linear predictor / growth advantage


def relative_fitness_dr_numpyro(
    seq_counts, N, dim, tau=None, pred=False, var_names=None
):
    N_time, N_variants = seq_counts.shape

    # Sample weights and latent factors
    phi_0 = numpyro.sample(
        "phi_0",
        TransformedDistribution(
            dist.Normal(0.0, 1.0).expand([dim]), OrderedTransform()
        ),
    )

    with numpyro.plate("N_dim_m1", dim):
        with numpyro.plate("N_variant_weight", N_variants - 1):
            _eta = numpyro.sample("eta", dist.Normal(0.0, 1.0))
            eta = jnp.vstack((_eta, jnp.zeros((1, dim))))
        with numpyro.plate("N_steps_base", N_time - 1):
            phi_rw_step = numpyro.sample("phi_rw_step", dist.Normal(0.0, 0.05))
            phi_rw = jnp.cumsum(phi_rw_step, axis=-1)

        # Combine increments and starting position
        phi_rw = jnp.vstack((jnp.zeros((1, dim)), phi_rw))
        phi = numpyro.deterministic("phi", softmax(phi_0 + phi_rw))

    # Compute fitness from weights and latent factors
    fitness = phi @ eta.T  # How do we ensure fitness of last variant is 0
    numpyro.deterministic("delta", fitness)

    # Sample initial frequency
    with numpyro.plate("N_variant_init", N_variants - 1):
        _init_logit = numpyro.sample("_init_logit", dist.Normal(0.0, 1.0)) * 5.0

    # Sum fitness to get dynamics over time
    init_logit = jnp.append(_init_logit, 0.0)
    logits = jnp.cumsum(fitness.at[0, :].set(0), axis=0) + init_logit

    # Evaluate likelihood
    obs = None if pred else np.nan_to_num(seq_counts)
    numpyro.sample(
        "seq_counts",
        dist.MultinomialLogits(logits=logits, total_count=np.nan_to_num(N)),
        obs=obs,
    )

    # Compute frequency
    numpyro.deterministic("freq", softmax(logits, axis=-1))

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(fitness[:, :-1] * tau)
        )  # Last row corresponds to linear predictor / growth advantage


class RelativeFitnessDR(ModelSpec):
    def __init__(
        self,
        dim: int,
        phi_model=None,
        tau: Optional[float] = None,
        hier: bool = False,
    ):
        self.dim = dim
        self.tau = tau
        self.hier = hier
        self.phi_model = phi_model if phi_model is not None else LatentRW()
        if self.hier:
            self.model_fn = partial(
                relative_fitness_dr_hier_numpyro,
                dim=self.dim,
                phi_model=self.phi_model,
            )
        else:
            self.model_fn = partial(relative_fitness_dr_numpyro, dim=self.dim)

    def augment_data(self, data: dict) -> None:
        data["dim"] = self.dim
        self.phi_model.build_model(data)
        return None

    def fit_mcmc(
        self,
        data: ef.VariantFrequencies,
        num_warmup: int = 100,
        num_samples: int = 100,
    ):
        """
        Abstract away NUTS stuff in Evofr and numpyro for quick usage.
        """
        inference_method = ef.InferNUTS(num_warmup=num_warmup, num_samples=num_samples)
        return inference_method.fit(self, data)
