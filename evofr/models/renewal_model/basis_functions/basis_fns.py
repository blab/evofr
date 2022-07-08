from abc import ABC, abstractmethod
from jax.interpreters.xla import DeviceArray

class BasisFunction(ABC):

    @abstractmethod
    def make_features(self, data: dict) -> DeviceArray:
        pass
