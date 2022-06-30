import jax.numpy as jnp
from evofr.data import expand_dates


def get_quantile(samples, p, site):
    q = jnp.array([0.5 * (1 - p), 0.5 * (1 + p)])
    return jnp.quantile(samples[site], q=q, axis=0)


def get_median(samples: dict, site):
    return jnp.median(samples[site], axis=0)


def get_quantiles(samples, ps, site):
    quants = []
    for i in range(len(ps)):
        quants.append(get_quantile(samples, ps[i], site))
    med = get_median(samples, site)
    return med, quants


def get_site_by_variant(samples, data, ps, name, site, forecast=False):

    # Unpack variant info
    var_names = data.var_names
    dates = data.dates

    # Unpack posterior
    site_name = site + "_forecast" if forecast else site
    site = samples[site_name]
    N_variant = site.shape[-1]
    T = site.shape[-2]

    if forecast:
        dates = expand_dates(dates, T)

    # Compute medians and hdis for ps
    site_median = jnp.median(site, axis=0)

    site_hdis = [
        jnp.quantile(site, q=jnp.array([0.5 * (1 - p), 0.5 * (1 + p)]), axis=0)
        for p in ps
    ]

    site_dict = dict()
    site_dict["date"] = []
    site_dict["location"] = []
    site_dict["variant"] = []
    site_dict[f"median_{site_name}"] = []
    for p in ps:
        site_dict[f"{site_name}_upper_{round(p * 100)}"] = []
        site_dict[f"{site_name}_lower_{round(p * 100)}"] = []

    for variant in range(N_variant):
        site_dict["date"] += list(dates)
        site_dict["location"] += [name] * T
        site_dict["variant"] += [var_names[variant]] * T
        site_dict[f"median_{site_name}"] += list(site_median[:, variant])
        for i, p in enumerate(ps):
            site_dict[f"{site_name}_upper_{round(ps[i] * 100)}"] += list(
                site_hdis[i][1, :, variant]
            )
            site_dict[f"{site_name}_lower_{round(ps[i] * 100)}"] += list(
                site_hdis[i][0, :, variant]
            )
    return site_dict


def get_freq(samples, LD, ps, name, forecast=False):
    return get_site_by_variant(
        samples, LD, ps, name, "freq", forecast=forecast
    )
