import pandas as pd
from evofr.data import CaseCounts
from evofr.data import HierCases

# TEST_CASES = pd.read_csv(
#     "../rt-from-frequency-dynamics/data/omicron-us-split/omicron-us-split_location-case-counts.tsv",
#     sep="\t",
# )
#
#
# def test_case_count():
#     filtered_test_cases = TEST_CASES[TEST_CASES.location == "Washington"].copy()
#     case_data = CaseCounts(filtered_test_cases)
#     print(case_data.dates)
#     print(case_data.make_data_dict())
#
#
# def test_hier_case_count():
#     hier_case_data = HierCases(TEST_CASES, group="location")
#     print(hier_case_data.make_data_dict())
