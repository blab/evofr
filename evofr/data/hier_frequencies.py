from typing import Optional

import numpy as np
import pandas as pd

from .data_helpers import format_var_names, prep_dates
from .data_spec import DataSpec
from .temporal_aggregation import aggregate_temporally_hierarchical
from .variant_frequencies import VariantFrequencies


class HierFrequencies(DataSpec):
    def __init__(
        self,
        raw_seq: pd.DataFrame,
        group: str,
        date_to_index: Optional[dict] = None,
        pivot: Optional[str] = None,
        aggregation_frequency: Optional[str] = None,
    ):
        """Construct a data specification for handling variant frequencies
        in hierarchical models.

        Parameters
        ----------
        raw_seq:
            a dataframe containing sequence counts with columns 'sequences',
            'variant', and date'.

        group:
            string defining which column to seperate data by.

        date_to_index:
            optional dictionary for mapping calender dates to nd.array indices.

        pivot:
            optional name of variant to place last.
            Defaults to "other" if present otherwise.
            This will usually used as a reference or pivot strain.

        aggregation_frequency:
            optional temporal frequency used to aggregate daily counts to
            larger time periods such as "W" (week) or "M" (month).

        Returns
        -------
        HierFrequencies
        """
        # Get date to index
        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_seq["date"])
        self.date_to_index = date_to_index

        # Get variant names
        raw_var_names = list(pd.unique(raw_seq.variant))
        raw_var_names.sort()
        self.var_names = format_var_names(raw_var_names, pivot=pivot)
        self.pivot = self.var_names[-1]

        # Loop each group
        grouped = raw_seq.groupby(group)
        self.names = [name for name, _ in grouped]
        self.groups = [
            VariantFrequencies(group, self.date_to_index, self.var_names)
            for _, group in grouped
        ]

        # Aggregate counts into larger windows
        self.aggregation_frequency = aggregation_frequency
        if self.aggregation_frequency is not None:
            (
                self.groups,
                self.dates,
                self.date_to_index,
            ) = aggregate_temporally_hierarchical(
                self.groups, self.dates, self.aggregation_frequency
            )

        self.seq_counts = np.stack([g.seq_counts for g in self.groups], axis=-1)

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()
        data["seq_counts"] = np.stack([g.seq_counts for g in self.groups], axis=-1)
        data["N"] = np.stack([g.seq_counts.sum(axis=-1) for g in self.groups], axis=-1)
        data["var_names"] = self.var_names
        return data
