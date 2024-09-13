from abc import ABC, abstractmethod


class BaseAlgo(ABC):
    def __init__(self, gamma: float):
        self.gamma = gamma
    
    @abstractmethod
    def _reset(self):
        ...