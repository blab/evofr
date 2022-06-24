import json
from typing import Dict, List, Optional

from evofr.data.data_spec import DataSpec


class PosteriorHandler:
    def __init__(
        self,
        dataset: Optional[Dict] = None,
        data: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ):
        self.dataset = dataset
        self.data = data
        self.name = name

    def save_posterior(self, filepath: str):
        if self.dataset is not None:
            with open(filepath, "w") as file:
                json.dump(self.dataset, file)

    def load_posterior(self, filepath: str):
        with open(filepath, "w") as file:
            self.dataset = json.load(file)

    def unpack_posterior(self):
        return self.dataset, self.data

    def get_site(self, site: str):
        if self.dataset:
            return self.dataset[site]

    def get_data_dict(self):
        if not self.data_dict and self.data:
            self.data_dict = self.data.make_data_dict()
        return self.data_dict


class MultiPosterior:
    def __init__(
        self,
        posteriors: Optional[List[PosteriorHandler]] = None,
        posterior: Optional[PosteriorHandler] = None,
    ):
        self.locator = dict()
        if posterior is not None:
            self.add_posterior(posterior)
        if posteriors is not None:
            self.add_posteriors(posteriors)

    def add_posterior(self, posterior: PosteriorHandler):
        self.locator[posterior.name] = posterior

    def add_posteriors(self, posteriors: List[PosteriorHandler]):
        for p in posteriors:
            self.add_posterior(p)

    def get(self, name):
        return self.locator[name]
