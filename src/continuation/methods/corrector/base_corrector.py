from abc import ABC, abstractmethod
from typing import Tuple


class Corrector(ABC):
    """Abstract Corrector to be inherited by developer for any new corrector."""

    @abstractmethod
    def correction_step(self) -> Tuple:
        pass

    @abstractmethod
    def _assign_states(self):
        pass
