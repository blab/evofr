from typing import Optional

import pandas as pd

from .data_helpers import prep_cases, prep_dates
from .data_spec import DataSpec


class CaseCounts(DataSpec):
    def __init__(self, raw_cases: pd.DataFrame, date_to_index: Optional[dict] = None):
        """Construct a data specification for handling case counts.

        Parameters
        ----------
        raw_cases:
            a dataframe containing case counts with columns 'cases' and 'date'.

        date_to_index:
            optional dictionary for mapping calender dates to nd.array indices.

        Returns
        -------
        CaseCounts
        """

        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_cases["date"])
        self.date_to_index = date_to_index

        self.cases = prep_cases(raw_cases, self.date_to_index)

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()
        data["cases"] = self.cases
        return data
