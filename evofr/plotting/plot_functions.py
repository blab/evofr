from typing import Callable, List, Optional
import jax.numpy as jnp

from evofr.posterior.posterior_helpers import get_median, get_quantiles


def prep_posterior_for_plot(
    site, samples, ps: List[float], forecast: Optional[bool] = False
):
    """Prep posteriors for plotting by finding time span, medians, and quantiles.

    Parameters
    ----------

    site:
        Name of the site to access from samples.

    samples:
        Dictionary with keys being site or variable names.
        Values are DeviceArrays containing posterior samples
        with shape (sample_number, site_shape).

    ps:
        Levels of confidence to generate quantiles for.

    forecast:
        Prep posterior for forecasts? Defaults to False.
    """
    med, quants = get_quantiles(samples, ps, site)
    t = jnp.arange(0, med.shape[0], 1)

    if forecast:
        med_f, quants_f = get_quantiles(samples, ps, site + "_forecast")
        t_f = med.shape[0] + jnp.arange(0, med_f.shape[0], 1)
        t = jnp.concatenate((t, t_f))
        med = jnp.concatenate((med, med_f))
        for i in range(len(ps)):
            quants[i] = jnp.concatenate([quants[i], quants_f[i]], axis=1)
    return t, med, quants


def plot_posterior_time(
    ax,
    t,
    med,
    quants,
    alphas: List[float],
    colors: List[str],
    included: Optional[List[bool]] = None,
):
    """
    Loop over variants to plot medians and quantiles at specifed points.
    Plots all time points unless time points to be included
    are specified in 'included'.

    Parameters
    ----------
    ax:
        Matplotlib axis to plot on.

    t:
        Time points to plot over.

    med:
        Median values.

    quants:
        Quantiles to be plotted. Organized as a list of CIs as DeviceArrays.

    alphas:
        Transparency for each quantile.

    colors:
        List of colors to use for each variant.

    included:
        optional list of bools which determine
        which time points and variants to include observations from.
    """

    for variant in range(med.shape[-1]):
        v_included = (
            jnp.arange(0, med.shape[0])
            if included is None
            else included[:, variant]
        )
        for i in range(len(quants)):
            ax.fill_between(
                t[v_included],
                quants[i][0, v_included, variant],
                quants[i][1, v_included, variant],
                color=colors[variant],
                alpha=alphas[i],
            )
        ax.plot(t[v_included], med[v_included, variant], color=colors[variant])


def plot_R(
    ax,
    samples,
    ps: List[float],
    alphas: List[float],
    colors: List[str],
    forecast: Optional[bool] = False,
):
    t, med, quants = prep_posterior_for_plot(
        "R", samples, ps, forecast=forecast
    )
    ax.axhline(y=1.0, color="k", linestyle="--")
    plot_posterior_time(ax, t, med, quants, alphas, colors)


def plot_R_censored(
    ax,
    samples,
    ps: List[float],
    alphas: List[float],
    colors: List[str],
    forecast: Optional[bool] = False,
    thres: Optional[float] = 0.001,
):
    t, med, quants = prep_posterior_for_plot(
        "R", samples, ps, forecast=forecast
    )
    ax.axhline(y=1.0, color="k", linestyle="--")

    # Plot only variants at high enough frequency
    _, freq_median, _ = prep_posterior_for_plot(
        "freq", samples, ps, forecast=forecast
    )
    included = freq_median > thres

    plot_posterior_time(ax, t, med, quants, alphas, colors, included=included)


def plot_posterior_average_R(
    ax, samples, ps: List[float], alphas: List[float], color: str
):
    med, V = get_quantiles(samples, ps, "R_ave")
    t = jnp.arange(0, V[-1].shape[-1], 1)

    # Make figure
    ax.axhline(y=1.0, color="k", linestyle="--")
    for i in range(len(ps)):
        ax.fill_between(
            t, V[i][0, :], V[i][1, :], color=color, alpha=alphas[i]
        )
    ax.plot(t, med, color=color)


def plot_little_r_censored(
    ax,
    samples,
    ps: List[float],
    alphas: List[float],
    colors: List[str],
    forecast: Optional[bool] = False,
    thres: Optional[float] = 0.001,
):
    t, med, quants = prep_posterior_for_plot(
        "r", samples, ps, forecast=forecast
    )
    ax.axhline(y=0.0, color="k", linestyle="--")

    # Plot only variants at high enough frequency
    _, freq_median, _ = prep_posterior_for_plot(
        "freq", samples, ps, forecast=forecast
    )
    included = freq_median > thres

    plot_posterior_time(ax, t, med, quants, alphas, colors, included=included)


def plot_posterior_frequency(
    ax,
    samples,
    ps: List[float],
    alphas: List[float],
    colors: List[str],
    forecast: Optional[bool] = False,
):
    t, med, quants = prep_posterior_for_plot(
        "freq", samples, ps, forecast=forecast
    )
    plot_posterior_time(ax, t, med, quants, alphas, colors)


def plot_observed_frequency(ax, LD, colors: List[str]):
    obs_freq = jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    N_variant = obs_freq.shape[-1]
    t = jnp.arange(0, obs_freq.shape[0])
    for variant in range(N_variant):
        ax.scatter(
            t, obs_freq[:, variant], color=colors[variant], edgecolor="black"
        )


def plot_observed_frequency_size(ax, LD, colors: List[str], size: Callable):
    N = LD.seq_counts.sum(axis=1)[:, None]
    sizes = [size(n) for n in N]
    obs_freq = jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    N_variant = obs_freq.shape[-1]
    t = jnp.arange(0, obs_freq.shape[0])
    for variant in range(N_variant):
        ax.scatter(
            t,
            obs_freq[:, variant],
            color=colors[variant],
            s=sizes,
            edgecolor="black",
        )


def plot_posterior_I(
    ax,
    samples,
    ps: List[float],
    alphas: List[float],
    colors: List[str],
    forecast: Optional[bool] = False,
):
    t, med, quants = prep_posterior_for_plot(
        "I_smooth", samples, ps, forecast=forecast
    )
    plot_posterior_time(ax, t, med, quants, alphas, colors)


def plot_posterior_smooth_EC(
    ax, samples, ps: List[float], alphas: List[float], color: str
):
    med, V = get_quantiles(samples, ps, "total_smooth_prev")
    t = jnp.arange(0, V[-1].shape[-1], 1)

    # Make figure
    for i in range(len(ps)):
        ax.fill_between(
            t, V[i][0, :], V[i][1, :], color=color, alpha=alphas[i]
        )
    ax.plot(t, med, color=color)


def plot_cases(ax, LD):
    t = jnp.arange(0, LD.cases.shape[0])
    ax.bar(t, LD.cases, color="black", alpha=0.3)


def add_dates(ax, dates, sep=1):
    t = []
    labels = []
    for (i, date) in enumerate(dates):
        if int(date.strftime("%d")) == 1:
            labels.append(date.strftime("%b"))
            t.append(i)
    ax.set_xticks(t[::sep])
    ax.set_xticklabels(labels[::sep])


def add_dates_sep(ax, dates, sep=7):
    t = []
    labels = []
    for (i, date) in enumerate(dates):
        if (i % sep) == 0:
            labels.append(date.strftime("%b %d"))
            t.append(i)
    ax.set_xticks(t)
    ax.set_xticklabels(labels)


def plot_growth_advantage(
    ax, samples, LD, ps: List[float], alphas: List[float], colors: List[str]
):
    ga = jnp.array(samples["ga"])

    inds = jnp.arange(0, ga.shape[-1], 1)

    ax.axhline(y=1.0, color="k", linestyle="--")
    parts = ax.violinplot(
        ga.T, inds, showmeans=False, showmedians=False, showextrema=False
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    q1, med, q3 = jnp.percentile(ga, jnp.array([25, 50, 75]), axis=0)
    ax.scatter(inds, med, color="white", zorder=3, edgecolor="black")
    ax.vlines(inds, q1, q3, color="k", lw=4, zorder=2)

    q1, med, q3 = jnp.percentile(ga, jnp.array([2.5, 50, 97.5]), axis=0)
    ax.vlines(inds, q1, q3, color="k", lw=2, zorder=1)

    ax.set_xticks(inds)
    ax.set_xticklabels(LD.var_names[:-1])


def plot_total_by_obs_frequency(ax, LD, total, colors: List[str]):
    T, D = LD.seq_counts.shape
    t = jnp.arange(0, T, 1)
    obs_freq = jnp.nan_to_num(
        jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    )

    # Make figure
    bottom = jnp.zeros(t.shape)
    for variant in range(D):
        ax.bar(
            t,
            obs_freq[:, variant] * total,
            bottom=bottom,
            color=colors[variant],
        )
        bottom = obs_freq[:, variant] * total + bottom


def plot_total_by_median_frequency(ax, samples, LD, total, colors: List[str]):
    T, D = LD.seq_counts.shape
    t = jnp.arange(0, T, 1)
    med_freq = get_median(samples, "freq")

    # Make figure
    bottom = jnp.zeros(t.shape)
    for variant in range(D):
        ax.bar(
            t,
            med_freq[:, variant] * total,
            bottom=bottom,
            color=colors[variant],
        )
        bottom = med_freq[:, variant] * total + bottom


def plot_ppc_frequency(
    ax,
    samples,
    LD,
    ps: List[float],
    alphas: List[float],
    colors: List[str],
    forecast: Optional[bool] = False,
):
    N = LD.seq_counts.sum(axis=-1)
    t, med, quants = prep_posterior_for_plot(
        "seq_counts", samples, ps, forecast=forecast
    )
    med = med / N[:, None]
    quants = [q / N[:, None] for q in quants]
    plot_posterior_time(ax, t, med, quants, alphas, colors)


def plot_ppc_seq_counts(
    ax,
    samples,
    ps: List[float],
    alphas: List[float],
    colors: List[str],
    forecast: Optional[bool] = False,
):
    t, med, quants = prep_posterior_for_plot(
        "seq_counts", samples, ps, forecast=forecast
    )
    plot_posterior_time(ax, t, med, quants, alphas, colors)


def plot_ppc_cases(
    ax, samples, ps: List[float], alphas: List[float], color: str
):
    med, V = get_quantiles(samples, ps, "cases")
    t = jnp.arange(0, V[-1].shape[-1], 1)

    # Make figure
    for i in range(len(ps)):
        ax.fill_between(
            t, V[i][0, :], V[i][1, :], color=color, alpha=alphas[i]
        )
    ax.plot(t, med, color=color)
