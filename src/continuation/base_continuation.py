from abc import ABC, abstractmethod


class Continuation(ABC):
    @abstractmethod
    def run(self):
        pass
