import logging
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
import sys
from src.DatasetProcesser import DatasetProcessor
import pandas as pd
from gensim.models import KeyedVectors


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
    """

    def __init__(self):
        self.active_dataset_name = None  # Name of the active dataset
        self.active_dataframe_name = None  # Name of the active dataframe
        self._datasets = {}  # Changed from list to dict for keyed access
        self._dataframes = {}
        self.word2vec_model = None

        logging.info("Dataset manager initialized.")

    def load_word2vec_model(self, data_path):
        """
        Loads a Word2Vec model from the specified data path.

        Args:
            data_path (str): The path to the Word2Vec model file.

        Returns:
            self: The DatasetManager instance.

        """
        self.word2vec_model = KeyedVectors.load_word2vec_format(data_path, binary=True)
        print("Word2Vec model loaded.")
        return self

    @property
    def active_dataset(self):
        if self.active_dataset_name and self.active_dataset_name in self._datasets:
            return self._datasets[self.active_dataset_name]
        else:
            logging.error("No active dataset set or dataset does not exist.")
            return None

    @property
    def active_dataframe(self):
        if (
            self.active_dataframe_name
            and self.active_dataframe_name in self._dataframes
        ):
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
        """
        Create a dataset from datasets load_dataset and load it from a CSV file.

        Args:
            name (str): The name of the dataset.
            data_path (str): The path to the CSV file.

        Returns:
            None

        Raises:
            Exception: If there is an error loading the dataset.

        """
        try:
            dataset = load_dataset("csv", data_files=data_path)
            self._datasets[name] = dataset
            if (
                not self.active_dataset_name
            ):  # Automatically set the first dataset as active
                self.active_dataset_name = name
            logging.info(f"Dataset loaded and set as {name}.")
        except Exception as e:
            logging.error(f"Failed to load dataset {name}: {e}")

    def load_dataframe(self, name, data_path, columns={0: "label", 1: "news"}):
        """
        Loads a dataframe from a CSV file.

        Args:
            name (str): The name of the dataframe.
            data_path (str): The path to the CSV file.
            columns (dict, optional): A dictionary mapping column indices to column names. Defaults to {0: "label", 1: "news"}.

        Returns:
            None
        """
        try:
            df = pd.read_csv(data_path, encoding="latin", header=True)
            df.rename(columns=columns, inplace=True)
            self._dataframes[name] = df
            if not self.active_dataframe_name:
                self.active_dataframe = name
            logging.info(f"Dataframe {name} loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Critical error loading dataframe: {e}")
            sys.exit("Cannot continue without dataframe.")
        except Exception as e:
            logging.error(f"Error loading dataframe: {e}")

    def load_processed(self,name,data_path):
        try:
            df = pd.read_csv(data_path, encoding="latin")
            self._dataframes[name] = df
            if not self.active_dataframe_name:
                self.active_dataframe = name
            logging.info(f"Dataframe {name} loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Critical error loading dataframe: {e}")
            sys.exit("Cannot continue without dataframe.")
        except Exception as e:
            logging.error(f"Error loading dataframe: {e}")





    def store_processed_dataset(self, name, processed_data):
        """
        Stores processed data within the manager.

        Args:
            name: The name of the dataset for identification.
            processed_data: The processed data to store, typically a DataFrame or a list of features.

        Returns:
            None
        """
        # If processed_data is not a DataFrame but a list (e.g., from feature extraction), convert it to DataFrame
        if not isinstance(processed_data, pd.DataFrame):
            processed_data = pd.DataFrame(processed_data)

        self._dataframes[name] = processed_data

    def from_dataframe(self, dataframe_name="main"):
        """
        Returns a dataset object from a dataframe

        Args:
            dataframe_name (str): The name of the dataframe to retrieve the dataset from. Defaults to "main".

        Returns:
            Dataset: The dataset object corresponding to the specified dataframe name. Returns None if the dataframe does not exist.
        """
        # Check if dataframe with given name exists
        if dataframe_name in self._dataframes:
            # Return dataset object
            return DatasetProcessor(self._dataframes[dataframe_name], manager=self)
        else:
            logging.error(f"Dataframe '{dataframe_name}' does not exist.")
            return None

    def get_dataset(self, name):
        """
        Retrieves a dataset by name.

        Args:
            name (str): The name of the dataset to retrieve.

        Returns:
            The dataset with the specified name.

        Raises:
            KeyError: If the dataset with the specified name is not found.
        """
        if name in self._datasets:
            return self._datasets[name]
        else:
            raise KeyError(f"Dataset '{name}' not found.")

    def __getitem__(self, name):
        """
        Retrieve a dataset by name.

        Args:
            name (str): The name of the dataset to retrieve.

        Returns:
            DatasetProcessor: The dataset processor for the specified dataset.

        Raises:
            KeyError: If the dataset with the specified name is not found.
        """
        if name in self._dataframes:
            return DatasetProcessor(self._dataframes[name])
        else:
            raise KeyError(f"Dataset '{name}' not found.")

    def get_financial_news(self):
        """This method is for demonstration of the deep learning process."""
        news = pd.read_csv("all-data.csv", encoding="latin", header=None)
        news.rename(columns={0: "label", 1: "news"}, inplace=True)
        return news

    def plot_external(self, df):
        """
            This method is for demonstration of the deep learning process.

            plot from external source
        Args:
            df (Dataframe): dataframe to plot

        Returns:
            None
        """
        df["label"].value_counts(ascending=True).plot.barh()
        plt.title("Balance of the target classes in the dataset")
        plt.show()
        print(df["label"].value_counts())

    def check_doc_len(self, df):
        """
        This method is for demonstration of the deep learning process.

        Args:
            param1 (type): Description of param1
            param2 (type): Description of param2

        Returns:
            type: Description of return value
        """

        df["len"] = df["news"].apply(lambda text: len(text.split()))
        return df["len"].max()
