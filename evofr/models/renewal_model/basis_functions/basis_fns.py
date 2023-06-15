from abc import ABC, abstractmethod
from jax import Array

class BasisFunction(ABC):

    @abstractmethod
    def make_features(self, data: dict) -> Array:
        pass
