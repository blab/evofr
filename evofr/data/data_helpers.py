import datetime
import numpy as np
import pandas as pd
from typing import List, Optional


def prep_dates(raw_dates: pd.Series):
    _dates = pd.to_datetime(raw_dates)
    dmn = _dates.min()
    dmx = _dates.max()
    n_days = (dmx - dmn).days
    dates = [dmn + datetime.timedelta(days=d) for d in range(0, 1 + n_days)]
    date_to_index = {d: i for (i, d) in enumerate(dates)}
    return dates, date_to_index


def forecast_dates(dates, T_forecast: int):
    last_date = dates[-1]
    dates_f = []
    for d in range(T_forecast):
        dates_f.append(last_date + datetime.timedelta(days=d + 1))
    return dates_f


def expand_dates(dates, T_forecast: int):
    x_dates = dates.copy()
    for d in range(T_forecast):
        x_dates.append(dates[-1] + datetime.timedelta(days=d + 1))
    return x_dates


def prep_cases(raw_cases: pd.DataFrame, date_to_index: Optional[dict] = None):
    raw_cases["date"] = pd.to_datetime(raw_cases["date"])
    if date_to_index is None:
        _, date_to_index = prep_dates(raw_cases["date"])

    T = len(date_to_index)
    C = np.full(T, np.nan)

    # Loop over dataframe rows to fill matrix
    for _, row in raw_cases.iterrows():
        C[date_to_index[row.date]] = row.cases
    return C


def format_var_names(raw_names: List[str]):
    """
    Places `other` category to be last element.
    """
    if "other" in raw_names:
        names = []
        for s in raw_names:
            if s != "other":
                names.append(s)
        names.append("other")
        return names
    return raw_names


def counts_to_matrix(
    raw_seqs: pd.DataFrame,
    var_names: List[str],
    date_to_index: Optional[dict] = None,
):
    """
    Turn `raw_seqs` to a matrix C with counts of each variant (v) on a given
    day t being element C[t,v]. Only uses names present in var_names.
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
):
    if var_names is None:
        raw_var_names = list(pd.unique(raw_seqs.variant))
        raw_var_names.sort()
        var_names = format_var_names(raw_var_names)
    C = counts_to_matrix(raw_seqs, var_names, date_to_index=date_to_index)
    return var_names, C
