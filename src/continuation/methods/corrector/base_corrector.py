from abc import ABC, abstractmethod
from typing import Tuple


class Corrector(ABC):
    @abstractmethod
    def correction_step(self) -> Tuple:
        pass

    @abstractmethod
    def assign_states(self):
        pass
