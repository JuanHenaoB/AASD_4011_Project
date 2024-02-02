from Preprocessor import Preprocessor
from FeatureExtractor import FeatureExtractor

class DatasetManager:
    def __init__(self):
        self.datasets = {}
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor()

    def load_dataset(self, name, data):
        # TODO: Implement dataset loading logic
        pass

    def get_dataset(self, name):
        # TODO: Implement dataset retrieval logic
        pass

    def set_dataset(self, name, dataset):
        # TODO: Implement dataset setting logic
        self.datasets[name] = dataset

    def preprocess(self, name):
        # TODO: Implement preprocessing logic
        pass

    def feature_extraction(self, name):
        # TODO: Implement feature extraction logic
        pass