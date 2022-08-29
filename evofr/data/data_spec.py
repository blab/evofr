from abc import ABC, abstractmethod
from typing import Optional


class DataSpec(ABC):
    @abstractmethod
    def make_data_dict(self, data: Optional[dict] = None) -> dict:
        """Get arguments to be passed to numpyro models as a dictionary.

        Parameters
        ----------
        data:
            optional dictionary to add arguments to.

        Returns
        --------
        data:
            dictionary containing arguments.
        """
        pass
