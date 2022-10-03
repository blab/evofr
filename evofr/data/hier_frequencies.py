from .variant_frequencies import VariantFrequencies
from .data_helpers import prep_dates, format_var_names
from .data_spec import DataSpec
import numpy as np
import pandas as pd
from typing import Optional


class HierFrequencies(DataSpec):
    def __init__(
        self,
        raw_seq: pd.DataFrame,
        group: str,
        date_to_index: Optional[dict] = None,
        pivot: Optional[str] = None,
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

        self.seq_counts = np.stack(
            [g.seq_counts for g in self.groups], axis=-1
        )

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()
        data["seq_counts"] = np.stack(
            [g.seq_counts for g in self.groups], axis=-1
        )
        data["N"] = np.stack(
            [g.seq_counts.sum(axis=-1) for g in self.groups], axis=-1
        )
        data["var_names"] = self.var_names
        return data
