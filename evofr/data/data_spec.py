from abc import ABC, abstractmethod
from typing import Optional


class DataSpec(ABC):
    @abstractmethod
    def make_data_dict(self, data: Optional[dict]=None) -> dict:
        pass
