import datetime
from typing import Dict, List, Optional

import numpy as np

from evofr.data import DataSpec


def add_dataspec_attr(export_dict, data, attr, key=None):
    """Add data specification attributes to dictionary."""
    key = key if key else attr
    if hasattr(data, attr):
        export_dict[key] = getattr(data, attr)


def get_quantiles(samples, ps, site):
    """Retrieve quantiles and median."""
    med = np.median(samples[site], axis=0)
    quants = [
        np.quantile(samples[site], [0.5 * (1 - p), 0.5 * (1 + p)], axis=0) for p in ps
    ]
    return med, quants


def forecast_dates(dates, T_forecast: int):
    """Generate dates of forecast given forecast interval of length 'T_forecast'."""
    last_date = dates[-1]
    return [last_date + datetime.timedelta(days=d + 1) for d in range(T_forecast)]


def create_entry(template, variant, value, ps, entries):
    """Append a new entry directly to the list to avoid intermediate copies."""
    entries.append(
        {
            **template,
            "variant": variant,
            "value": np.around(value, decimals=3),
            "ps": ps,
        }
    )


def process_site(
    entries,
    samples,
    ps,
    site,
    date_map,
    variants,
    location,
    forecast_date_map=None,
):
    med, quants = get_quantiles(samples, ps, site)

    for v, variant in enumerate(variants):
        if v >= med.shape[1]:
            continue
        template = {"site": site, "variant": variant, "location": location}

        relevant_date_map = (
            forecast_date_map if forecast_date_map else (date_map if date_map else None)
        )
        if relevant_date_map:
            for date, index in relevant_date_map.items():
                template["date"] = date.strftime("%Y-%m-%d")
                create_entry(template, variant, med[index, v], "median", entries)

                for i, p in enumerate(ps):
                    create_entry(
                        template,
                        variant,
                        quants[i][0, index, v],
                        f"HDI_{round(p * 100)}_lower",
                        entries,
                    )
                    create_entry(
                        template,
                        variant,
                        quants[i][1, index, v],
                        f"HDI_{round(p * 100)}_upper",
                        entries,
                    )
        else:
            # Handle non-dated sites differently if needed
            create_entry(template, variant, med[v], "median", entries)
            for i, p in enumerate(ps):
                create_entry(
                    template,
                    variant,
                    quants[i][0, v],
                    f"HDI_{round(p * 100)}_lower",
                    entries,
                )
                create_entry(
                    template,
                    variant,
                    quants[i][1, v],
                    f"HDI_{round(p * 100)}_upper",
                    entries,
                )


def get_sites_variants_tidy(samples, data, sites, dated, forecasts, ps, name=None):
    """Generate tidy data and metadata for given sites and samples."""
    # Initialize metadata and configure probability keys
    ps_keys = (
        ["median"]
        + [f"HDI_{round(p * 100)}_upper" for p in ps]
        + [f"HDI_{round(p * 100)}_lower" for p in ps]
    )
    metadata = {"ps": ps_keys, "sites": sites, "location": name}

    add_dataspec_attr(metadata, data, "dates", "dates")
    add_dataspec_attr(metadata, data, "var_names", "variants")

    # Prepare entries and handle all cases
    entries = []
    date_map = data.date_to_index if dated else None

    for site, is_dated, is_forecast in zip(sites, dated, forecasts):
        forecast_date_map = None
        if is_forecast and is_dated:
            T = samples[site].shape[1]
            forecasted_dates = forecast_dates(data.dates, T)
            forecast_date_map = {d: i for (i, d) in enumerate(forecasted_dates)}
            metadata["forecast_dates"] = forecasted_dates

        process_site(
            entries,
            samples,
            ps,
            site,
            date_map if is_dated else None,
            metadata["variants"],
            name,
            forecast_date_map if is_forecast else None,
        )
    return {"metadata": metadata, "data": entries}
