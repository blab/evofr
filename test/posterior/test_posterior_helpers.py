from evofr.posterior.posterior_helpers import combine_sites_tidy


def test_combine_sites_tidy():
    tidy_dicts = [
        {
            "metadata": {
                "location": ["Africa"],
                "ps_point_estimator": "mean",
            },
            "data": [
                {
                    "record": 1,
                }
            ],
        },
        {
            "metadata": {
                "location": ["Europe"],
                "ps_point_estimator": "mean",
            },
            "data": [
                {
                    "record": 2,
                }
            ],
        }
    ]
    combined_dict = combine_sites_tidy(tidy_dicts)
    assert sorted(combined_dict["metadata"]["location"]) == ["Africa", "Europe"]
    assert combined_dict["metadata"]["ps_point_estimator"] == "mean"
    assert len(combined_dict["data"]) == 2
