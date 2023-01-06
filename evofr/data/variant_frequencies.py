from .data_helpers import prep_dates, prep_sequence_counts
from .data_spec import DataSpec
import pandas as pd
from typing import List, Optional

VARIANT_NAMES = ["Variant", "other"]
START_DATE = pd.to_datetime("2022-01-01")


def variant_counts_to_dataframe(
    var_counts,
    var_names: List[str] = VARIANT_NAMES,
    start_date=START_DATE,
):
    """Convert matrix of variant counts to pandas dataframe
    for input to ef.VariantFrequencies.

    Parameters
    ----------
    var_counts:
        nd.array of counts var_counts[t,v] of variant v on day t.
    variant_names:
        List of variant names to assign each column.
    start_date:
        Pandas datetime to use as first date.

    Returns
    -------
    seq_counts
    """
    T, V = var_counts.shape

    assert V <= len(var_names), "More cols present in var_counts than names"

    sequences = []
    variant = []
    dates = []

    for time in range(T):
        # Add wildtype and variant counts
        sequences.extend(list(var_counts[time, :]))

        # Add variant labels
        variant.extend(var_names[:V])

        # Add dates
        dates.extend([start_date + pd.to_timedelta(time, unit="d")] * V)

    df = pd.DataFrame(
        {
            "variant": variant,
            "sequences": sequences,
            "date": dates,
        }
    )

    return df[df.sequences > 0].reset_index(drop=True)


class VariantFrequencies(DataSpec):
    def __init__(
        self,
        raw_seq: pd.DataFrame,
        date_to_index: Optional[dict] = None,
        var_names: Optional[List] = None,
        pivot: Optional[str] = None,
    ):
        """Construct a data specification for handling variant frequencies.

        Parameters
        ----------
        raw_seq:
            a dataframe containing sequence counts with columns 'sequences',
            'variant', and date'.

        date_to_index:
            optional dictionary for mapping calender dates to nd.array indices.

        var_names:
            optional list containing names of variants to be present.

        pivot:
            optional name of variant to place last.
            Defaults to "other" if present otherwise.
            This will usually used as a reference or pivot strain.

        Returns
        -------
        VariantFrequencies
        """

        # Get mapping from date to index
        if date_to_index is None:
            self.dates, date_to_index = prep_dates(raw_seq["date"])
        self.date_to_index = date_to_index

        # Turn dataframe to counts of each variant sequenced each day
        self.var_names, self.seq_counts = prep_sequence_counts(
            raw_seq, self.date_to_index, var_names, pivot=pivot
        )
        self.pivot = self.var_names[-1]

    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        if data is None:
            data = dict()

        # Export `seq_counts`, sequences each day, and variant names
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        data["var_names"] = self.var_names
        return data
