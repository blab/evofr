import datetime
import numpy as np
import pandas as pd
from typing import List, Optional


def prep_dates(raw_dates: pd.Series):
    """Return vector of dates and a mapping of dates to indices.

    Parameters
    ----------
    raw_dates:
        pandas series containing dates of interest

    Returns
    -------
    dates:
        list containing dates

    date_to_index:
        dictionary taking in dates and returning integer indices
    """
    _dates = pd.to_datetime(raw_dates)
    dmn = _dates.min()
    dmx = _dates.max()
    n_days = (dmx - dmn).days
    dates = [dmn + datetime.timedelta(days=d) for d in range(0, 1 + n_days)]
    date_to_index = {d: i for (i, d) in enumerate(dates)}
    return dates, date_to_index


def forecast_dates(dates, T_forecast: int):
    """Generate dates of forecast given forecast interval of length 'T_forecast'."""
    last_date = dates[-1]
    dates_f = []
    for d in range(T_forecast):
        dates_f.append(last_date + datetime.timedelta(days=d + 1))
    return dates_f


def expand_dates(dates, T_forecast: int):
    """Extend existing dates list with forecast interval of length 'T_forecast'"""
    x_dates = dates.copy()
    for d in range(T_forecast):
        x_dates.append(dates[-1] + datetime.timedelta(days=d + 1))
    return x_dates


def prep_cases(raw_cases: pd.DataFrame, date_to_index: Optional[dict] = None):
    """Process raw_cases data to nd.array including unobserved dates.

    Parameters
    ----------
    raw_cases:
        a dataframe containing case counts with columns 'cases' and 'date'.

    date_to_index:
        optional dictionary for mapping calender dates to nd.array indices.

    Returns
    -------
    C:
        nd.array containing number of cases on each date.
    """
    raw_cases["date"] = pd.to_datetime(raw_cases["date"])
    if date_to_index is None:
        _, date_to_index = prep_dates(raw_cases["date"])

    T = len(date_to_index)
    C = np.full(T, np.nan)

    # Loop over dataframe rows to fill matrix
    for _, row in raw_cases.iterrows():
        C[date_to_index[row.date]] = row.cases
    return C


def format_var_names(raw_names: List[str], pivot: Optional[str] = None):
    """Places pivot category to be last element if present."""
    pivot = "other" if pivot is None else pivot
    if pivot in raw_names:
        # Move pivot the end
        names = []
        for s in raw_names:
            if s != pivot:
                names.append(s)
        names.append(pivot)
        return names
    # Otherwise, return the original names
    return raw_names


def counts_to_matrix(
    raw_seqs: pd.DataFrame,
    var_names: List[str],
    date_to_index: Optional[dict] = None,
):
    """Process 'raw_seq' data to nd.array including unobserved dates.

    Parameters
    ----------
    raw_seq:
        a dataframe containing sequence counts with
        columns 'sequences' and 'date'.

    var_names:
        list of variant to count observations.

    date_to_index:
        optional dictionary for mapping calender dates to nd.array indices.

    Returns
    -------
    C:
        nd.array containing number of sequences of each variant on each date.
    """
    raw_seqs["date"] = pd.to_datetime(raw_seqs["date"])
    if date_to_index is None:
        _, date_to_index = prep_dates(raw_seqs["date"])

    T = len(date_to_index)
    V = len(var_names)
    C = np.full((T, V), np.nan)

    # Loop over rows in dataframe to fill matrix
    for i, s in enumerate(var_names):
        for _, row in raw_seqs[raw_seqs.variant == s].iterrows():
            C[date_to_index[row.date], i] = row.sequences

    # Loop over matrix rows to correct for zero counts
    for d in range(T):
        if not np.isnan(C[d, :]).all():
            # If not all days samples are nan, replace nan with zeros
            np.nan_to_num(C[d, :], copy=False)
    return C


def prep_sequence_counts(
    raw_seqs: pd.DataFrame,
    date_to_index: Optional[dict] = None,
    var_names: Optional[List] = None,
    pivot: Optional[str] = None,
):
    """Process 'raw_seq' data to nd.array including unobserved dates.

    Parameters
    ----------
    raw_seq:
        a dataframe containing sequence counts with
        columns 'sequences' and 'date'.


    date_to_index:
        optional dictionary for mapping calender dates to nd.array indices.

    var_names:
        optional list of variant to count observations.

    pivot:
        optional name of variant to place last.
        This will usually used as a reference or pivot strain.

    Returns
    -------
    var_names:
        list of variants counted

    C:
        nd.array containing number of sequences of each variant on each date.
    """

    if var_names is None:
        raw_var_names = list(pd.unique(raw_seqs.variant))
        raw_var_names.sort()
        var_names = format_var_names(raw_var_names, pivot=pivot)
    C = counts_to_matrix(raw_seqs, var_names, date_to_index=date_to_index)
    return var_names, C
