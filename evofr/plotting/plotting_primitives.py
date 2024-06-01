from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_site_in_time(
    ax,
    samples: dict,
    site: str,
    dates: Optional[list[str | pd.Timestamp]] = None,
    colors: Optional[list[str]] = None,
    quantiles: Optional[list[float]] = None,
    alphas: Optional[list[float]] = None,
):
    """Plots a single time series with optional quantiles. Accepts an existing ax."""
    values = samples[site]
    times = pd.to_datetime(dates) if dates is not None else np.arange(values.shape[1])

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    if quantiles:
        # Plot requested quantiles
        median = np.median(values, axis=0)
        ax.plot(times, median, color=colors)

        alphas = [1.0] * len(quantiles) if alphas is None else alphas
        for alpha, quantile in zip(alphas, quantiles):
            lower = np.quantile(values, 0.5 * (1 - quantile), axis=0)
            upper = np.quantile(values, 0.5 * (1 + quantile), axis=0)
            ax.fill_between(times, lower, upper, color=colors, alpha=alpha)
    else:
        # Plot all samples
        alphas = [1.0] if alphas is None else alphas
        for i in range(values.shape[0]):
            ax.plot(times, values[i, ...], color=colors, alpha=alphas[0])
    return ax


def plot_variants(
    ax,
    samples: dict,
    site: str,
    variants: list[str],
    variants_to_plot: Optional[list[str]] = None,
    plot_type: str = "violin",
    color_map: Optional[dict[str, str]] = None,
    quantiles: Optional[list[float]] = None,
):
    """Plots multiple variants as either violin plots, histograms, or quantile intervals. Accepts an existing ax."""
    values = samples[site]
    variants_to_plot = variants if variants_to_plot is None else variants_to_plot

    if color_map is None:
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, values.shape[1]))
    else:
        colors = [color_map[v] for v in variants_to_plot]

    variant_indices = [variants.index(v) for v in variants_to_plot]

    if plot_type == "violin":
        sns.violinplot(
            data=values[:, variant_indices], palette=colors, scale="width", ax=ax
        )
    elif plot_type == "histogram":
        for color, idx, variant in zip(colors, variant_indices, variants_to_plot):
            sns.histplot(
                values[:, idx],
                color=color,
                alpha=0.5,
                bins=20,
                label=variant,
                ax=ax,
            )
        ax.set_xlabel(site)
    elif plot_type == "quantiles":
        assert quantiles is not None
        for idx, variant in enumerate(variants_to_plot):
            variant_values = values[:, variant_indices[idx]]

            median = np.median(variant_values, axis=0)
            for quantile in quantiles:
                lower = np.quantile(variant_values, 0.5 * (1 - quantile), axis=0)
                upper = np.quantile(variant_values, 0.5 * (1 + quantile), axis=0)

                # Plot quantiles as error bars
                ax.errorbar(
                    x=idx,
                    y=median,
                    yerr=[[median - lower], [upper - median]],
                    fmt="o",
                    color=colors[idx],
                    label=variant,
                )
            ax.scatter([idx], [median], color=colors[idx])
    if plot_type == "violin" or plot_type == "quantiles":
        ax.set_xticks(range(len(variants_to_plot)))
        ax.set_xticklabels(variants_to_plot)
    return ax


def plot_time_series_with_variants(
    ax,
    samples: dict,
    site: str,
    variants: list[str],
    dates: Optional[list[str | pd.Timestamp]] = None,
    variants_to_plot: Optional[list[str]] = None,
    color_map: Optional[dict[str, str]] = None,
    quantiles: Optional[list[float]] = None,
    alphas: Optional[list[float]] = None,
):
    """Plots a time series with variants with optional quantiles. Accepts an existing ax."""
    values = samples[site]
    times = pd.to_datetime(dates) if dates is not None else np.arange(values.shape[1])
    variants_to_plot = variants if variants_to_plot is None else variants_to_plot

    if color_map is None:
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, values.shape[2]))
    else:
        colors = [color_map[v] for v in variants_to_plot]

    for idx, variant in enumerate(variants_to_plot):
        variant_values = values[:, :, idx]
        # Plot requested quantiles
        if quantiles:
            median = np.median(variant_values, axis=0)
            ax.plot(times, median, color=colors[idx], label=variant)

            # Draw shaded intervals for quantiles
            alphas = [1.0] * len(quantiles) if alphas is None else alphas
            for alpha, quantile in zip(alphas, quantiles):
                lower = np.quantile(variant_values, 0.5 * (1 - quantile), axis=0)
                upper = np.quantile(variant_values, 0.5 * (1 + quantile), axis=0)
                ax.fill_between(times, lower, upper, color=colors[idx], alpha=alpha)
        else:
            # Plot all samples
            alphas = [1.0] if alphas is None else alphas
            for sample in variant_values:
                ax.plot(times, sample, color=colors[idx], alpha=alphas[0])

    ax.set_xlabel("Date" if dates is not None else "Time Step")
    ax.set_ylabel("Values")
    if quantiles:
        ax.legend()
