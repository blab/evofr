from abc import ABC, abstractmethod
from typing import Optional


class DataSpec(ABC):
    # Registry to hold all subclasses by their class name.
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        DataSpec.registry[cls.__name__] = cls

    @abstractmethod
    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        """
        Get arguments to be passed to numpyro models as a dictionary.
        
        Parameters
        ----------
        data:
            Optional dictionary to add arguments to.
        
        Returns
        -------
        dict:
            Dictionary containing arguments.
        """
        pass
