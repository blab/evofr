import json
import pickle
from typing import Dict, List, Optional

from evofr.data.data_spec import DataSpec
from evofr.posterior.posterior_helpers import EvofrEncoder


def determine_method(filepath):
    """
    Determines the serialization method based on the file extension.

    Parameters:
        filepath (str): The path of the file including its extension.

    Returns:
        str: The serialization method ("json" or "pickle").

    Raises:
        ValueError: If the file extension is not recognized.
    """
    import os

    _, ext = os.path.splitext(filepath)
    if ext.lower() == ".json":
        return "json"
    elif ext.lower() == ".pkl":
        return "pickle"
    return None


def save_data(data, filename, method="json"):
    """
    Save data to a file using either JSON or pickle based on the user's choice.

    Parameters:
    - data: The data to be serialized and saved.
    - filename: The filename where the data will be saved.
    - method: The serialization method ('json' or 'pickle').

    Raises:
    - ValueError: If the provided method is not supported.
    """
    if method == "json":
        with open(filename, "w") as file:
            json.dump(data, file, cls=EvofrEncoder)
    elif method == "pickle":
        with open(filename, "wb") as file:
            pickle.dump(data, file)
    else:
        raise ValueError("Unsupported serialization method. Use 'json' or 'pickle'.")


def load_data(filename, method="json"):
    """
    Load data from a file using either JSON or pickle based on the user's choice.

    Parameters:
    - filename: The filename from which the data will be loaded.
    - method: The serialization method ('json' or 'pickle').

    Returns:
    - The data loaded from the file.

    Raises:
    - ValueError: If the provided method is not supported.
    """
    if method == "json":
        with open(filename, "r") as file:
            return json.load(file)
    elif method == "pickle":
        with open(filename, "rb") as file:
            return pickle.load(file)
    else:
        raise ValueError("Unsupported deserialization method. Use 'json' or 'pickle'.")


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
            Values will be Arrays containing posterior samples.

        data:
            optional DataSpec instance containing underlying data from analysis

        name:
            name of the posterior. Used to index in MultiPosterior.
        """
        self.samples = samples if samples else dict()
        self.data = data
        self.name = name

    def save_posterior(self, filepath: str, method=None):
        if method is None:
            method = determine_method(filepath)
            assert (
                method is not None
            ), """Serialization method could not be determined from `filepath`. 
                Please define explicitly or use compatiable file extension e.g. .json or .pkl)"""

        """Save posterior samples at a given filepath."""
        save_data(self.samples, filepath, method=method)

    def load_posterior(self, filepath: str, method=None):
        """Load posterior samples from a given filepath."""
        if method is None:
            method = determine_method(filepath)
            assert (
                method is not None
            ), """Serialization method could not be determined from `filepath`. 
                Please define explicitly or use compatiable file extension e.g. .json or .pkl)"""
        self.samples = load_data(filepath, method=method)
        return self

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
