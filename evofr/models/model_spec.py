from abc import ABC, abstractmethod
from typing import Callable


class ModelSpec(ABC):
    model_fn: Callable

    @abstractmethod
    def augment_data(self, data: dict) -> None:
        pass
