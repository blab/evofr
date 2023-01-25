import pandas as pd
from evofr.data import VariantFrequencies
from evofr.data.hier_frequencies import HierFrequencies

TEST_SEQ = pd.read_csv(
    "./test/testing_data/mlr-variant-counts.tsv",
    sep="\t",
)


def test_variant_frequencies():
    filtered_test_seq = TEST_SEQ[TEST_SEQ.location == "City0"].copy()
    seq_data = VariantFrequencies(filtered_test_seq, pivot="C")
    data = seq_data.make_data_dict()

    assert data["N"].shape == (70,)
    assert data["seq_counts"].shape == (70, 3)


def test_hier_variant_frequencies():
    hier_seq_data = HierFrequencies(TEST_SEQ, group="location", pivot="C")
    data = hier_seq_data.make_data_dict()

    assert data["seq_counts"].shape == (70, 3, 2)
    assert data["N"].shape == (70, 2)
