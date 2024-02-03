from pandas import DataFrame

import numpy as np
import pandas as pd
import logging
import sys

class Dataset:
    
    def __init__(self):
        self.dataframe: dataframe = None
        self.features: [] = None

    def load_dataframe(self, data_path):
        try:
            self.dataframe = pd.read_csv(data_path, encoding='latin', header=None)
            logging.info(f"dataframe loaded from file.")
        
        except FileNotFoundError as e:
            logging.error(f"Critical Error loading dataframe: {e}")
            sys.exit("Cannot continue without dataframe.")

        except Exception as e:
            logging.error(f"Error loading dataframe: {str(e)}")

    def get_features(self):
        # TODO: Implement feature retrieval logic
        return self.features

    def set_features(self, features):
        # TODO: Implement feature setting logic
        self.features = features

    def get_dataframe(self):
        return self.dataframe
        

    def get_features(self):
        # TODO: Implement feature retrieval logic
        return self.features

    def set_features(self, features):
        # TODO: Implement feature setting logic
        self.features = features
