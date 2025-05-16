import pandas as pd

from evofr.data.data_helpers import forecast_dates


def test_forecast_dates():
    """The interval in days between forecast dates should reflect the interval
    between the input dates.

    """
    # In this first example, we use daily inputs.
    dates = [pd.to_datetime(date) for date in ["2020-01-01", "2020-01-02"]]
    future_dates = [date.strftime("%Y-%m-%d") for date in forecast_dates(dates, 1)]
    assert future_dates == ['2020-01-03']

    # In the next example, we use inputs separated by 14 days.
    dates = [pd.to_datetime(date) for date in ["2020-01-01", "2020-01-15"]]
    future_dates = [date.strftime("%Y-%m-%d") for date in forecast_dates(dates, 1)]
    assert future_dates == ['2020-01-29']

    # When only a single input date is given, we have to assume an interval of 1 day.
    dates = [pd.to_datetime(date) for date in ["2020-01-15"]]
    future_dates = [date.strftime("%Y-%m-%d") for date in forecast_dates(dates, 1)]
    assert future_dates == ['2020-01-16']
