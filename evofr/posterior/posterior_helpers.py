from typing import Dict, List, Optional
import jax.numpy as jnp
import json
import numpy as np
import pandas as pd
from evofr.data import forecast_dates
from evofr.data import DataSpec


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


def get_site_by_variant(
    samples: Dict, data: DataSpec, ps, name, site, forecast=False
):

    # Unpack variant info
    var_names = data.var_names
    dates = data.dates

    # Unpack posterior
    site_name = site + "_forecast" if forecast else site
    site = samples[site_name]
    N_variant = site.shape[-1]
    T = site.shape[-2]

    if forecast:
        dates = forecast_dates(dates, T)

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


def get_freq(samples: Dict, data: DataSpec, ps, name, forecast=False):
    return get_site_by_variant(
        samples, data, ps, name, "freq", forecast=forecast
    )


def get_growth_advantage(samples, data, ps, name, rel_to="other"):
    # Unpack variant info
    var_names = data.var_names

    # Get posterior samples
    ga = samples["ga"]
    ga = jnp.concatenate((ga, jnp.ones(ga.shape[0])[:, None]), axis=1)
    N_variant = ga.shape[-1]

    # Loop over ga and make relative rel_to
    for i, s in enumerate(var_names):
        if s == rel_to:
            ga = jnp.divide(ga, ga[:, i][:, None])

    # Compute medians and quantiles
    meds = jnp.median(ga, axis=0)
    gas = []
    for i, p in enumerate(ps):
        up = 0.5 + p / 2
        lp = 0.5 - p / 2
        gas.append(jnp.quantile(ga, jnp.array([lp, up]), axis=0).T)

    # Make empty dictionary
    v_dict = dict()
    v_dict["location"] = []
    v_dict["variant"] = []
    v_dict["median_ga"] = []

    for p in ps:
        v_dict[f"ga_upper_{round(p * 100)}"] = []
        v_dict[f"ga_lower_{round(p * 100)}"] = []

    for variant in range(N_variant):
        if var_names[variant] != rel_to:
            v_dict["location"].append(name)
            v_dict["variant"].append(var_names[variant])
            v_dict["median_ga"].append(meds[variant])
            for i, p in enumerate(ps):
                v_dict[f"ga_upper_{round(p * 100)}"].append(gas[i][variant, 1])
                v_dict[f"ga_lower_{round(p * 100)}"].append(gas[i][variant, 0])

    return v_dict


class EvofrEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, jnp.DeviceArray):
            return self.default(np.array(obj))
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        return json.JSONEncoder.default(self, obj)


def get_sites_quantiles_json(
    samples: Dict,
    data: DataSpec,
    sites: List[str],
    ps,
    name: Optional[str] = None,
):
    export_dict = dict()

    # Save common attributes at highest level
    export_dict["ps"] = ps
    export_dict["sites"] = sites
    if name:
        export_dict["name"] = name

    # Names from dataspec
    def add_dataspec_attr(
        export_dict: dict, data, attr: str, key: Optional[str] = None
    ) -> None:
        key = key if key else attr
        if hasattr(data, attr):
            export_dict[key] = getattr(data, attr)
        return None

    add_dataspec_attr(export_dict, data, "dates", key="dates")
    add_dataspec_attr(export_dict, data, "var_names", key="variants")

    # Each site has sub-dict with its info
    for site in sites:
        site_dict = dict()
        site_samples = samples[f"{site}"]

        # Get median
        site_dict["median"] = jnp.median(site_samples, axis=0)

        # Get ps
        for i, p in enumerate(ps):
            q = jnp.array(
                [0.5 * (1 - p), 0.5 * (1 + p)]
            )  # Upper and lower bound
            site_dict[f"HDI_{round(ps[i] * 100)}"] = jnp.quantile(
                site_samples,
                q=q,
                axis=0,
            )

        # Make site dict in dict
        export_dict[site] = site_dict
    return export_dict


def get_sites_variants_json(
    samples: Dict,
    data: DataSpec,
    sites: List[str],
    ps,
    name: Optional[str] = None,
):
    export_dict = dict()

    # Save common attributes at highest level

    # Make keys for probability levels
    ps_keys = ["median"]
    for i, p in enumerate(ps):
        ps_keys.append(f"HDI_{round(ps[i] * 100)}_upper")
        ps_keys.append(f"HDI_{round(ps[i] * 100)}_lower")
    export_dict["ps"] = ps_keys

    export_dict["sites"] = sites
    if name:
        export_dict["location"] = name

    # Names from dataspec
    def add_dataspec_attr(
        export_dict: dict, data, attr: str, key: Optional[str] = None
    ) -> None:
        key = key if key else attr
        if hasattr(data, attr):
            export_dict[key] = getattr(data, attr)
        return None

    add_dataspec_attr(export_dict, data, "dates", key="dates")
    add_dataspec_attr(export_dict, data, "var_names", key="variants")
    variants = export_dict["variants"]

    # Each site has sub-dict with its info
    for site in sites:
        site_dict = dict()
        site_samples = samples[f"{site}"]

        for v, variant in enumerate(variants):
            variant_dict = dict()

            # Get median
            variant_dict["median"] = jnp.median(
                site_samples[:, v, ...], axis=0
            )

            # Get ps
            for i, p in enumerate(ps):
                q = jnp.array(
                    [0.5 * (1 - p), 0.5 * (1 + p)]
                )  # Upper and lower bound

                # Get HDI Upper
                variant_dict[f"HDI_{round(ps[i] * 100)}_upper"] = jnp.quantile(
                    site_samples,
                    q=q[1],
                    axis=0,
                )

                # Get HDI Lower
                variant_dict[f"HDI_{round(ps[i] * 100)}_lower"] = jnp.quantile(
                    site_samples,
                    q=q[0],
                    axis=0,
                )
            site_dict[variant] = variant_dict

        # Make site dict in dict
        export_dict[site] = site_dict
    return export_dict


def save_json(out: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(out, f, cls=EvofrEncoder)
    return None
