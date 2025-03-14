from abc import ABC, abstractmethod
from typing import Callable


class ModelSpec(ABC):
    """Abstract model class.
    Used by evofr to handle model specifications for inference.
    Classes which inherit from ModelSpec must have an attribute 'model_fn'
    which defines the function to be used for inference in numpyro.
    """
    registry = {}

    model_fn: Callable

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Automatically register the subclass under its class name.
        ModelSpec.registry[cls.__name__] = cls

    @abstractmethod
    def augment_data(self, data: dict) -> None:
        """
        Augments existing data for inference with model specific information.
        """
        pass
