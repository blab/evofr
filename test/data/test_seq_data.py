import pandas as pd
from evofr.data import VariantFrequencies
from evofr.data.hier_frequencies import HierFrequencies

TEST_SEQ = pd.read_csv(
    "../rt-from-frequency-dynamics/data/omicron-us-split/omicron-us-split_location-variant-sequence-counts.tsv",
    sep="\t",
)


def test_variant_frequencies():
    filtered_test_seq = TEST_SEQ[TEST_SEQ.location == "Washington"].copy()
    seq_data = VariantFrequencies(filtered_test_seq)
    print(seq_data.make_data_dict())


def test_hier_variant_frequencies():
    hier_seq_data = HierFrequencies(TEST_SEQ, group="location")
    data = hier_seq_data.make_data_dict()

    assert data["seq_counts"].shape == (138, 7, 23)
    assert data["N"].shape == (138, 23)
