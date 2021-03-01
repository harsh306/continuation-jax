from abc import ABC, abstractmethod


class Continuation(ABC):
    """Abstract Continuation Strategy to be inherited by developer for any new Continuation Strategy."""

    @abstractmethod
    def run(self):
        pass
