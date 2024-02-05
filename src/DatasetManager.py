import logging
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
import sys
import pandas as pd


class DatasetManager:
    """
    Manages the loading, storing, and accessing of datasets for analysis and processing.
    
    This class provides methods to load datasets from various sources, manage them in memory,
    and access them for data processing tasks. It supports both datasets loaded as pandas DataFrames
    and datasets loaded from the Hugging Face 'datasets' library.
    
    Attributes
    ----------
    active_dataset_name : str
        The name of the currently active dataset loaded from the 'datasets' library.
    active_dataframe_name : str
        The name of the currently active DataFrame.
    _datasets : dict
        A dictionary storing datasets loaded from the 'datasets' library, keyed by dataset name.
    _dataframes : dict
        A dictionary storing pandas DataFrames, keyed by dataframe name.
    
    Methods
    -------
    create_dataset(name: str, data_path: str):
        Loads a dataset from a specified path and stores it under the given name.
    load_dataframe(name, data_path, columns={0: 'label', 1: 'news'}):
        Loads a CSV file as a pandas DataFrame, optionally renaming columns, and stores it.
    check_balance(label_key='label'):
        Plots the distribution of values in the specified label column of the active dataframe.
    store_processed_dataset(name, processed_data):
        Stores processed data, converting it to a DataFrame if necessary.
    
    Examples
    --------
    >>> dataset_manager = DatasetManager()
    >>> dataset_manager.load_dataframe('financial_news', 'path/to/financial_news.csv')
    >>> dataset_manager.check_balance('label')
    """
    
    def __init__(self):
        self.active_dataset_name = None  # Name of the active dataset
        self.active_dataframe_name = None  # Name of the active dataframe
        self._datasets = {}  # Changed from list to dict for keyed access
        self._dataframes = {}
        logging.info("Dataset manager initialized.")

    @property
    def active_dataset(self):
        if self.active_dataset_name and self.active_dataset_name in self._datasets:
            return self._datasets[self.active_dataset_name]
        else:
            logging.error("No active dataset set or dataset does not exist.")
            return None

    @property
    def active_dataframe(self):
        if self.active_dataframe_name and self.active_dataframe_name in self._dataframes:
            return self._dataframes[self.active_dataframe_name]
        else:
            logging.error("No active dataframe set or dataframe does not exist.")
            return None

    @active_dataframe.setter
    def active_dataframe(self, name):
        if name in self._dataframes:
            self.active_dataframe_name = name
            logging.info(f"Active dataframe set to {name}.")
        else:
            logging.error("Dataframe name does not exist.")

    def create_dataset(self, name: str, data_path: str):
        try:
            dataset = load_dataset('csv', data_files=data_path)
            self._datasets[name] = dataset
            if not self.active_dataset_name:  # Automatically set the first dataset as active
                self.active_dataset_name = name
            logging.info(f"Dataset loaded and set as {name}.")
        except Exception as e:
            logging.error(f"Failed to load dataset {name}: {e}")

    def load_dataframe(self, name, data_path, columns={0: 'label', 1: 'news'}):
        try:
            df = pd.read_csv(data_path, encoding='latin', header=None)
            df.rename(columns=columns, inplace=True)
            self._dataframes[name] = df
            if not self.active_dataframe_name:  # Automatically set the first dataframe as active
                self.active_dataframe = name  # Using the setter to ensure consistency
            logging.info(f"Dataframe {name} loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Critical error loading dataframe: {e}")
            sys.exit("Cannot continue without dataframe.")
        except Exception as e:
            logging.error(f"Error loading dataframe: {e}")

    def check_balance(self, label_key='label'):
        if self.active_dataframe is not None:
            self.active_dataframe[label_key].value_counts(ascending=True).plot.barh()
            plt.title("Balance of the target classes in the dataset")
            plt.show()
            print(self.active_dataframe[label_key].value_counts())
        else:
            logging.error("Attempted to check balance without an active dataframe set.")


    def store_processed_dataset(self, name, processed_data):
        """
        Stores processed data within the manager.

        Parameters:
        - name: The name of the dataset for identification.
        - processed_data: The processed data to store, typically a DataFrame or a list of features.
        """
        # If processed_data is not a DataFrame but a list (e.g., from feature extraction), convert it to DataFrame
        if not isinstance(processed_data, pd.DataFrame):
            processed_data = pd.DataFrame(processed_data)
        
        self._dataframes[name] = processed_data
