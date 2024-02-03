import numpy as np
import pandas as pd
import logging
import sys

class Dataset:
    
    def __init__(self):
        self.dataset = None
        self.features = None

    def load_data(self, data_path):
        try:
            self.datasets = pd.read_csv(data_path, encoding='latin', header=None)
            logging.info(f"Dataset loaded from file.")
        
        except FileNotFoundError as e:
            logging.error(f"Critical Error loading dataset: {e}")
            sys.exit("Cannot continue without dataset.")

        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")

    def get_features(self):
        # TODO: Implement feature retrieval logic
        return self.features

    def set_features(self, features):
        # TODO: Implement feature setting logic
        self.features = features


        

    def get_features(self):
        # TODO: Implement feature retrieval logic
        return self.features

    def set_features(self, features):
        # TODO: Implement feature setting logic
        self.features = features
