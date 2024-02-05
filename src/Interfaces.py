from abc import ABC, abstractmethod

class PreprocessingStep(ABC):
    @abstractmethod
    def apply(self, data):
        pass

class FeatureExtractionStep(ABC):
    @abstractmethod
    def extract(self, data):
        pass
