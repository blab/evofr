from typing import Dict, Optional, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from evofr.data.data_helpers import expand_dates
from evofr.data.data_spec import DataSpec
from evofr.plotting.plot_functions import (
    add_dates_sep,
    plot_R_censored,
    plot_cases,
    plot_growth_advantage,
    plot_observed_frequency,
    plot_posterior_I,
    plot_posterior_frequency,
    plot_ppc_frequency,
)

from evofr.posterior.posterior_handler import PosteriorHandler


DEFAULT_FIG_SIZE = (10, 6)
DEFAULT_PS = [0.8]
DEFAULT_ALPHAS = [0.4]


def create_empty_gridspec(
    nrows: int, ncols: int, figsize: Optional[Tuple] = None
):
    fig = plt.figure(
        figsize=figsize if figsize is not None else DEFAULT_FIG_SIZE
    )
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    return fig, gs


def create_date_axis(ax, plot_dates, date_sep=None, forecast_L=0):
    if forecast_L > 0:
        plot_dates = expand_dates(plot_dates, forecast_L)
        ax.axvline(
            x=len(plot_dates) - 1, color="k", linestyle="--"
        )  # Adding forecast cut off
    sep = date_sep if date_sep is not None else 20
    add_dates_sep(ax, plot_dates, sep=sep)  # Adding dates
    return None


class EvofrPlot:
    def __init__(
        self,
        posterior: Optional[PosteriorHandler] = None,
        samples: Optional[Dict] = None,
        data: Optional[DataSpec] = None,
    ):
        if posterior is not None:
            self.samples = posterior.samples
            self.data = posterior.data
        else:
            self.samples = samples if samples is not None else dict()
            self.data = data if data is not None else None


class FrequencyPlot(EvofrPlot):
    def __init__(
        self,
        posterior: Optional[PosteriorHandler] = None,
        samples: Optional[Dict] = None,
        data: Optional[DataSpec] = None,
        color_map: Optional[dict] = None,
    ):
        super().__init__(posterior=posterior, samples=samples, data=data)
        self.color_map = color_map if color_map is not None else dict()

    def plot(
        self,
        ax=None,
        figsize: Optional[Tuple] = None,
        forecast: Optional[bool] = False,
        date_sep: Optional[int] = None,
        forecast_L: int = 0,
        posterior: bool = True,
        observed: bool = True,
        predictive: bool = False,
    ):
        if ax is None:
            # Create a figure and axis
            fig, gs = create_empty_gridspec(1, 1, figsize=figsize)
            ax = fig.add_subplot(gs[0])

        colors = [
            self.color_map[v] for v in self.data.var_names
        ]  # Mapping colors to observed variants

        # Plot predicted frequencies
        if posterior:
            plot_posterior_frequency(
                ax,
                self.samples,
                DEFAULT_PS,
                DEFAULT_ALPHAS,
                colors,
                forecast=forecast,
            )

        if predictive:
            plot_ppc_frequency(
                ax,
                self.samples,
                self.data,
                DEFAULT_PS,
                DEFAULT_ALPHAS,
                colors,
            )

        if observed:
            plot_observed_frequency(
                ax, self.data, colors
            )  # Plot observed frequencies

        if hasattr(self.data, "dates"):
            create_date_axis(ax, self.data.dates, date_sep, forecast_L)
        ax.set_ylabel("Variant frequency")  # Making ylabel
        return ax


class GrowthAdvantagePlot(EvofrPlot):
    def __init__(
        self,
        posterior: Optional[PosteriorHandler] = None,
        samples: Optional[Dict] = None,
        data: Optional[DataSpec] = None,
        color_map: Optional[dict] = None,
    ):
        super().__init__(posterior=posterior, samples=samples, data=data)
        self.color_map = color_map if color_map is not None else dict()

    def plot(
        self,
        ax=None,
        figsize: Optional[Tuple] = None,
    ):
        # TODO: Add support for time-varying growth advantages
        if ax is None:
            # Create a figure and axis
            fig, gs = create_empty_gridspec(1, 1, figsize=figsize)
            ax = fig.add_subplot(gs[0])

        colors = [
            self.color_map[v] for v in self.data.var_names
        ]  # Mapping colors to observed variants

        plot_growth_advantage(
            ax, self.samples, self.data, DEFAULT_PS, DEFAULT_ALPHAS, colors
        )
        ax.set_ylabel("Growth advantage")
        return ax


class RtPlot(EvofrPlot):
    def __init__(
        self,
        posterior: Optional[PosteriorHandler] = None,
        samples: Optional[Dict] = None,
        data: Optional[DataSpec] = None,
        color_map: Optional[dict] = None,
        thres: Optional[float] = None,
    ):
        super().__init__(posterior=posterior, samples=samples, data=data)
        self.color_map = color_map if color_map is not None else dict()
        self.thres = thres if thres is not None else 0.01

    def plot(
        self,
        ax=None,
        figsize: Optional[Tuple] = None,
        date_sep: Optional[int] = None,
    ):
        # TODO: Add option to remove dashed line at 1.0
        if ax is None:
            # Create a figure and axis
            fig, gs = create_empty_gridspec(1, 1, figsize=figsize)
            ax = fig.add_subplot(gs[0])

        colors = [
            self.color_map[v] for v in self.data.var_names
        ]  # Mapping colors to observed variants

        plot_R_censored(
            ax,
            self.samples,
            DEFAULT_PS,
            DEFAULT_ALPHAS,
            colors,
            thres=0.001,
        )
        ax.set_ylabel("Effective Reproduction number")  # Making ylabel
        if hasattr(self.data, "dates"):
            create_date_axis(ax, self.data.dates, date_sep)

        return ax


class IncidencePlot(EvofrPlot):
    def __init__(
        self,
        posterior: Optional[PosteriorHandler] = None,
        samples: Optional[Dict] = None,
        data: Optional[DataSpec] = None,
        color_map: Optional[dict] = None,
    ):
        super().__init__(posterior=posterior, samples=samples, data=data)
        self.color_map = color_map if color_map is not None else dict()

    def plot(
        self,
        ax=None,
        figsize: Optional[Tuple] = None,
        cases: Optional[bool] = True,
        date_sep: Optional[int] = None,
    ):
        if ax is None:
            # Create a figure and axis
            fig, gs = create_empty_gridspec(1, 1, figsize=figsize)
            ax = fig.add_subplot(gs[0])

        colors = [
            self.color_map[v] for v in self.data.var_names
        ]  # Mapping colors to observed variants

        # Plot posterior variant specific incidence
        plot_posterior_I(
            ax,
            self.samples,
            DEFAULT_PS,
            DEFAULT_ALPHAS,
            colors,
        )

        if cases:
            plot_cases(ax, self.data)
        if hasattr(self.data, "dates"):
            create_date_axis(ax, self.data.dates, date_sep)
        return ax


class PatchLegend:
    def __init__(
        self,
        color_map: Dict,
        ncol: Optional[int] = None,
        loc: Optional[str] = None,
    ):
        self.color_map = color_map
        self.ncol = ncol
        self.loc = loc

    def add_legend(self, fig=None, ax=None):
        if fig is None and ax is not None:
            fig = ax.get_figure()
        patches = [
            mpl.patches.Patch(color=color, label=label)
            for label, color in self.color_map.items()
        ]

        ncol = len(self.color_map.keys()) if self.ncol is None else self.ncol
        labels = list(self.color_map.keys())
        legend = fig.legend(patches, labels, ncol=ncol, loc=self.loc)
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_edgecolor("k")
        return fig
