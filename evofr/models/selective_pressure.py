import jax.numpy as jnp

from evofr import PosteriorHandler


def compute_selective_pressure(
    posterior: PosteriorHandler,
    fitness_site: str = "delta",
    frequency_site: str = "freq",
    fixed: bool = False,
):
    # Unpack fitness and frequency
    fitness = posterior.samples[fitness_site]
    freq = posterior.samples[frequency_site]

    # Make sure frequency and fitness have the same shape
    if fitness.shape != freq.shape:
        if fixed:  # Extend to proper time-dim fitness
            T = freq.shape[1]
            fitness = jnp.repeat(jnp.expand_dims(fitness, axis=1), repeats=T, axis=1)

        num_samples, T = fitness.shape[0], fitness.shape[1]
        fitness = jnp.concatenate((fitness, jnp.zeros((num_samples, T, 1))), axis=-1)

    # Mean fitness
    delta_bar = jnp.mean(fitness * freq, axis=-1, keepdims=True)

    # Mean square deviation
    delta_sse = jnp.square(fitness - delta_bar)
    selective_pressure_var = (delta_sse * freq).sum(axis=-1)

    # Change in relative fitness overall
    fitness_change = jnp.diff(fitness, axis=1, prepend=jnp.nan)
    selective_pressure_expect = (fitness_change * freq).sum(axis=-1)
    selective_pressure = selective_pressure_var + selective_pressure_expect

    return delta_bar, selective_pressure


def compute_selective_pressure_hier(
    posterior: PosteriorHandler,
    fitness_site: str = "delta",
    frequency_site: str = "freq",
    fixed: bool = False,
):
    # Unpack fitness and frequency
    fitness = posterior.samples[fitness_site]
    freq = posterior.samples[frequency_site]

    # Make sure frequency and fitness have the same shape
    if fitness.shape != freq.shape:
        if fixed:  # Extend to proper time-dim fitness
            T = freq.shape[1]
            fitness = jnp.repeat(jnp.expand_dims(fitness, axis=1), repeats=T, axis=1)

        num_samples, T, _, num_groups = fitness.shape
        fitness = jnp.concatenate(
            (fitness, jnp.zeros((num_samples, T, 1, num_groups))), axis=-2
        )

    # Mean fitness
    delta_bar = jnp.sum(fitness * freq, axis=-2, keepdims=True)

    # Mean square deviation
    delta_sse = jnp.square(fitness - delta_bar)
    selective_pressure_var = (delta_sse * freq).sum(axis=-2)

    # Change in relative fitness overall
    fitness_change = jnp.diff(fitness, axis=1, prepend=jnp.nan)
    selective_pressure_expect = (fitness_change * freq).sum(axis=-1)
    selective_pressure = selective_pressure_var + selective_pressure_expect

    return delta_bar, selective_pressure


def compute_selective_pressure_general(
    posterior: PosteriorHandler,
    fitness_site: str = "delta",
    frequency_site: str = "freq",
    fixed: bool = False,
    hierarchical: bool = False,
    variance_only: bool = False,
):
    """
    Calculate selective pressures based on fitness and frequency estimates from a posterior distribution.
    This function accommodates both hierarchical and non-hierarchical data structures and computes selective pressures
    as a combination of variance in fitness and expected change in fitness over time.

    Parameters:
    - posterior (PosteriorHandler): An object containing the posterior samples. It must include
      estimates for fitness and frequency in the form of numpy arrays.
    - fitness_site (str, optional): The key in the `posterior.samples` dictionary that corresponds to
      the fitness estimates. Defaults to "delta".
    - frequency_site (str, optional): The key in the `posterior.samples` dictionary that corresponds to
      the frequency estimates. Defaults to "freq".
    - fixed (bool, optional): If True, converts fixed fitness estimates into corresponding time-varying estimates
      to match the frequency data's time dimension. Defaults to False.
    - hierarchical (bool, optional): Indicates whether the data includes hierarchical groupings
      (e.g., different populations or experimental groups). If True, operations account for an additional grouping dimension.
      Defaults to False.
    - variance_only (bool, optional): If True, use only the variance term for the selective pressure.

    Returns:
    - delta_bar (numpy.ndarray): An array of the mean fitness values, calculated as the weighted average
      of fitness across the variant dimension, retaining the hierarchical structure if applicable.
      Shape is (N, T, G) if hierarchical or (N, T) if not hierarchical.
    - selective_pressure (numpy.ndarray): An array representing the total selective pressure, computed as
      the sum of the variance of fitness and the expected change in fitness over time.
      Shape is (N, T, G) if hierarchical or (N, T) if not hierarchical.
    """

    # Unpack fitness and frequency
    fitness = posterior.samples[fitness_site]
    freq = posterior.samples[frequency_site]

    # Adjust fitness dimensions if fixed
    if fitness.shape != freq.shape:
        if fixed or len(fitness.shape) != len(freq.shape):
            T = freq.shape[1]
            fitness = jnp.repeat(jnp.expand_dims(fitness, axis=1), repeats=T, axis=1)

    # Pad relative fitness if number of variants differ
    if fitness.shape[2] != freq.shape[2]:
        num_samples, T = fitness.shape[0], fitness.shape[1]
        if hierarchical:
            num_groups = fitness.shape[-1]
            fitness = jnp.concatenate(
                (fitness, jnp.zeros((num_samples, T, 1, num_groups))), axis=-2
            )
        else:
            fitness = jnp.concatenate(
                (fitness, jnp.zeros((num_samples, T, 1))), axis=-1
            )

    variant_axis = -2 if hierarchical else -1

    # Calculate mean fitness
    delta_bar = jnp.sum(fitness * freq, axis=variant_axis, keepdims=True)

    # Calculate mean square deviation
    delta_sse = jnp.square(fitness - delta_bar)
    selective_pressure_var = (delta_sse * freq).sum(axis=variant_axis)

    # Calculate change in relative fitness
    fitness_change = jnp.diff(fitness, axis=1, prepend=jnp.nan)
    selective_pressure_expect = (fitness_change * freq).sum(variant_axis)
    if variance_only:
        selective_pressure = selective_pressure_var
    else:
        selective_pressure = selective_pressure_var + selective_pressure_expect

    delta_bar = jnp.squeeze(delta_bar, axis=variant_axis)
    return delta_bar, selective_pressure
