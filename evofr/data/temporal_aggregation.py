import pandas as pd


def aggregate_temporally(seq_counts, dates, frequency):
    """
    Aggregates time-series data based on a specified frequency (e.g., weekly, monthly).

    Parameters:
        - 'seq_counts' (numpy.ndarray): A 2D array where each row corresponds to a time point and columns
        - 'dates' (list of pandas.Timestamp): A list of timestamps corresponding to each row in 'seq_counts'.

    - frequency (str): A string representing the frequency of aggregation, according to pandas offset aliases.
        Examples include 'W-SUN' for weekly aggregation ending on Sunday, 'M' for monthly.

    Returns:
        - 'seq_counts_agg' (numpy.ndarray): A 2D array where each row corresponds to aggregated counts
        - 'dates_agg' (list of pandas.Timestamp): A list of timestamps corresponding to each row in 'seq_counts'.
        - 'date_to_index' (dict): A dictionary mapping timestamps to row in 'seq_counts_agg'

    """
    columns_seq_counts = [f"seq_{i}" for i in range(seq_counts.shape[1])]
    df = pd.DataFrame(seq_counts, index=dates, columns=columns_seq_counts)

    # Grouping the data according to the specified frequency
    grouped = df.groupby(pd.Grouper(freq=frequency)).sum()

    seq_counts_agg = grouped[columns_seq_counts].values
    dates_agg = list(grouped.index)
    date_to_index = {d: i for (i, d) in enumerate(dates_agg)}
    return seq_counts_agg, dates_agg, date_to_index


def aggregate_temporally_hierarchical(groups, dates, frequency):
    """
    Applies `aggregate_temporally` to each group within a hierarchical model.
    """
    for group in groups:
        seq_counts, dates_agg, date_to_index = aggregate_temporally(
            group.seq_counts, dates, frequency
        )
        group.seq_counts = seq_counts
        group.dates = dates_agg
        group.date_to_index = date_to_index

    return groups, dates_agg, date_to_index
