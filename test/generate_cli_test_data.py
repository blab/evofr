#!/usr/bin/env python
import os

import numpy as np
import pandas as pd


def simulate_MLR_freq(growth_advantage, freq0, tau, max_time):
    """Simulate variant frequencies over time."""
    times = np.arange(max_time)
    delta = np.log(growth_advantage) / tau
    ufreq = freq0 * np.exp(delta * times[:, None])
    return ufreq / ufreq.sum(axis=-1)[:, None]


def simulate_MLR(growth_advantage, freq0, tau, Ns):
    """Simulate sequence counts using multinomial draws."""
    max_time = len(Ns)
    freq = simulate_MLR_freq(growth_advantage, freq0, tau, max_time)
    seq_counts = [
        np.random.multinomial(int(Ns[t]), freq[t, :]) for t in range(max_time)
    ]
    return freq, np.stack(seq_counts)


def variant_counts_to_dataframe(
    var_counts, var_names, start_date=pd.to_datetime("2022-01-01")
):
    """
    Convert a matrix of variant counts to a tidy DataFrame.
    Only rows with positive counts are kept.
    """
    T, V = var_counts.shape
    sequences, variants, dates = [], [], []
    for t in range(T):
        sequences.extend(list(var_counts[t, :]))
        variants.extend(var_names[:V])
        dates.extend([start_date + pd.Timedelta(days=t)] * V)
    df = pd.DataFrame(
        {
            "variant": variants,
            "sequences": sequences,
            "date": dates,
        }
    )
    return df[df.sequences > 0].reset_index(drop=True)


def expand_seq_counts(df, location="TestCity"):
    """
    Expand aggregated sequence counts into strain-level metadata.

    For each row in the aggregated DataFrame, repeat it 'sequences' times.
    Add a 'location' column (if not present) and rename 'variant' to 'clade'.
    The output DataFrame has columns: location, clade, date.
    """
    df = df.copy()
    if "location" not in df.columns:
        df["location"] = location
    expanded = df.loc[df.index.repeat(df["sequences"])].reset_index(drop=True)
    expanded = expanded.drop(columns=["sequences"])
    expanded = expanded.rename(columns={"variant": "clade"})
    expanded = expanded[["location", "clade", "date"]]
    return expanded


def generate_test_data():
    os.makedirs("data", exist_ok=True)

    # --- Simulate Sequence Counts ---
    deltas = np.array([1.1, 1.8, 1.0])
    freq0 = np.array([4e-2, 1e-5, 0.999])
    freq0 = freq0 / freq0.sum()
    Ns = 50 * np.ones(100)  # 100 days

    _, var_counts = simulate_MLR(deltas, freq0, tau=4.2, Ns=Ns)
    seq_counts = variant_counts_to_dataframe(var_counts, var_names=["A", "B", "C"])
    seq_counts["clade"] = seq_counts["variant"]
    seq_counts.to_csv("data/processed_sequences_test.tsv", sep="\t", index=False)
    print("Saved test/data/processed_sequences_test.tsv")

    # Expand to strain-level metadata
    # raw_sequences = expand_seq_counts(seq_counts, location="TestCity")
    # Save as tests/data/raw_sequences.tsv (note: no "_expanded" suffix)
    seq_counts.to_csv("data/raw_sequences.tsv", sep="\t", index=False)
    print("Saved test/data/raw_sequences.tsv")

    # --- Simulate Case Counts ---
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    np.random.seed(42)
    cases = np.random.poisson(lam=400, size=len(dates))
    cases_df = pd.DataFrame({"location": "TestCity", "date": dates, "cases": cases})
    cases_df.to_csv("data/raw_cases.tsv", sep="\t", index=False)
    print("Saved test/data/raw_cases.tsv")


if __name__ == "__main__":
    generate_test_data()
