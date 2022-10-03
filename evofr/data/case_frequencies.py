from .data_helpers import prep_dates, format_var_names
from .data_spec import DataSpec
from .case_counts import CaseCounts
from .variant_frequencies import VariantFrequencies
import numpy as np
import pandas as pd
from typing import List, Optional


class CaseFrequencyData(DataSpec):
    def __init__(
        self,
        raw_cases: pd.DataFrame,
        raw_seq: pd.DataFrame,
        date_to_index: Optional[dict] = None,
        var_names: Optional[List] = None,
        pivot: Optional[str] = None,
    ):
        """Construct a data specification for handling case counts
        and variant frequencies.

        Parameters
        ----------
        raw_cases:
            a dataframe containing case counts with columns 'cases' and 'date'.

        raw_seq:
            a dataframe containing sequence counts with columns 'sequences',
            'variant', and date'.

        date_to_index:
            optional dictionary for mapping calender dates to nd.array indices.

        var_names:
            optional list containing names of variants to be present

        pivot:
            optional name of variant to place last.
            Defaults to "other" if present otherwise.
            This will usually used as a reference or pivot strain.

        Returns
        -------
        CaseFrequencyData
        """

        # Get dates
        raw_dates = pd.concat((raw_cases["date"], raw_seq["date"]))
        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_dates)
        self.date_to_index = date_to_index

        # Get cases with CaseData
        case_data = CaseCounts(raw_cases, self.date_to_index)
        self.cases = case_data.cases

        # Get sequence data with VariantFrequencies
        self.pivot = pivot
        freq_data = VariantFrequencies(
            raw_seq, self.date_to_index, var_names, self.pivot
        )
        self.seq_counts = freq_data.seq_counts
        self.var_names = freq_data.var_names

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()

        data["cases"] = self.cases
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        data["var_names"] = self.var_names
        return data


class HierarchicalCFData:
    def __init__(
        self,
        raw_cases: pd.DataFrame,
        raw_seq: pd.DataFrame,
        group: str,
        date_to_index: Optional[dict] = None,
    ):
        """Construct a data specification for handling case counts
        and variant frequencies in a hierarchical model.

        Parameters
        ----------
        raw_cases:
            a dataframe containing case counts with columns 'cases' and 'date'.

        raw_seq:
            a dataframe containing sequence counts with columns 'sequences',
            'variant', and date'.

        data_to_index:
            optional dictionary for mapping calender dates to nd.array indices.

        group:
            string defining which column to seperate data by.

        Returns
        -------
        HierarchicalCFData
        """
        # Get dates
        raw_dates = raw_cases["date"].append(raw_seq["data"])
        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_dates)
        self.date_to_index = date_to_index

        # Get variant names
        raw_var_names = list(pd.unique(raw_seq.variant))
        raw_var_names.sort()
        self.var_names = format_var_names(raw_var_names)

        # Loop each group
        gseq = raw_seq.groupby(group)
        gcase = raw_cases.groupby(group)
        self.names = [name for name, _ in gseq]
        self.groups = [
            CaseFrequencyData(
                gcase.groups[name],
                gseq.groups[name],
                self.date_to_index,
                self.var_names,
            )
            for name in self.names
        ]

    def make_data_dict(self, data: Optional[dict] = None):
        if data is None:
            data = dict()

        data["cases"] = np.stack([g.cases for g in self.groups], axis=-1)
        data["seq_counts"] = np.stack(
            [g.seq_counts for g in self.groups],
            axis=-1,
        )
        data["N"] = np.stack(
            [g.seq_counts.sum(axis=-1) for g in self.groups], axis=-1
        )
        data["var_names"] = self.var_names
        return data
