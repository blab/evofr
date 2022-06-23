from .case_counts import CaseCounts
from .data_helpers import prep_dates
from .data_spec import DataSpec
import numpy as np
import pandas as pd
from typing import Optional


class HierCases(DataSpec):
    def __init__(
        self,
        raw_cases: pd.DataFrame,
        group: str,
        date_to_index: Optional[dict] = None,
    ):
        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_cases["date"])
        self.date_to_index = date_to_index

        # Loop each group
        grouped = raw_cases.groupby(group)
        self.names = [name for name, _ in grouped]
        self.groups = [
            CaseCounts(group, self.date_to_index)
            for _, group in grouped
        ]

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()
        data["cases"] = np.stack(
            [g.cases for g in self.groups],
            axis=-1
        )
        return data
