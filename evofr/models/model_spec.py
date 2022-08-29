from abc import ABC, abstractmethod
from typing import Callable


class ModelSpec(ABC):
    """Abstract model class.
    Used by evofr to handle model specifications for inference.
    Classes which inherit from ModelSpec must have an attribute 'model_fn'
    which defines the function to be used for inference in numpyro.
    """
    model_fn: Callable

    @abstractmethod
    def augment_data(self, data: dict) -> None:
        """
        Augments existing data for inference with model specific information.

        Parameters
        ----------
        data:
            dictionary which contains arguments to 'model_fn'

        Returns
        -------
        None
        """
        pass
