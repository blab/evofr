import json
from typing import Dict, List, Optional

from evofr.data.data_spec import DataSpec
from evofr.posterior.posterior_helpers import EvofrEncoder


class PosteriorHandler:
    def __init__(
        self,
        samples: Optional[Dict] = None,
        data: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ):
        """Construct PosteriorHandler.

        Parameters
        ----------
        samples:
            optional dictionary with keys corresponding to variable names.
            Values will be DeviceArrays containing posterior samples.

        data:
            optional DataSpec instance containing underlying data from analysis

        name:
            name of the posterior. Used to index in MultiPosterior.
        """
        self.samples = samples if samples else dict()
        self.data = data
        self.name = name

    def save_posterior(self, filepath: str):
        """Save posterior samples at a given filepath."""
        if self.samples is not None:
            with open(filepath, "w") as file:
                json.dump(self.samples, file, cls=EvofrEncoder)

    def load_posterior(self, filepath: str):
        """Load posterior samples from a given filepath."""
        with open(filepath, "w") as file:
            self.samples = json.load(file)

    def unpack_posterior(self):
        """Return samples and dataspec from PosteriorHandler."""
        return self.samples, self.data

    def get_site(self, site: str):
        """Get samples at a given site from PosteriorHandler."""
        if self.samples:
            return self.samples[site]

    def get_data_dict(self):
        """Convert dataspec to corresponding dictionary."""
        if not self.data_dict and self.data:
            self.data_dict = self.data.make_data_dict()
        return self.data_dict


class MultiPosterior:
    def __init__(
        self,
        posteriors: Optional[List[PosteriorHandler]] = None,
        posterior: Optional[PosteriorHandler] = None,
    ):
        """Construct MultiPosterior.

        Parameters
        ----------
        posteriors:
            optional list of PosteriorHandlers to be added

        posterior:
            optional PosteriorHandler to be added to object.
        """
        self.locator = dict()
        if posterior is not None:
            self.add_posterior(posterior)
        if posteriors is not None:
            self.add_posteriors(posteriors)

    def add_posterior(self, posterior: PosteriorHandler):
        """Add PosteriorHandler to MultiPosterior by its name."""
        self.locator[posterior.name] = posterior

    def add_posteriors(self, posteriors: List[PosteriorHandler]):
        """Add multiple PosteriorHandlers to MultiPosterior by name."""
        for p in posteriors:
            self.add_posterior(p)

    def get(self, name):
        """Return PosteriorHandler by name"""
        return self.locator[name]

    def __getitem__(self, name: str):
        return self.get(name)
