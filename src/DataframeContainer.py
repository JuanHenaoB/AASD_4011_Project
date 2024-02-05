from pandas import DataFrame

import numpy as np
import pandas as pd
import logging
import sys

class DataframeContainer:
    """
        A DataframeContainer class, contains a dataframe and features for the dataframe
    
        Attributes
        ----------
        dataframe : Dataframe
            Description of the attribute.
    
        Methods
        -------
        method_name
            Description of the method.
        
        Examples
        --------
        Examples of how to use this class.
    """
    
    def __init__(self):
        self.dataframe: DataFrame = None
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


    @property
    def dataframe(self):
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self,dataframe):
        self._dataframe = dataframe

    
    @dataframe.getter
    def dataframe(self):
        return self._dataframe

    def get_features(self):
        # TODO: Implement feature retrieval logic
        return self.features

    def set_features(self, features):
        # TODO: Implement feature setting logic
        self.features = features
