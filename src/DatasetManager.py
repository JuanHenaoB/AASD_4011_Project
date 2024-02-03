from src.Preprocessor import Preprocessor
from src.FeatureExtractor import FeatureExtractor
import logging
from src.Dataset import Dataset


class DatasetManager:
    """
    # TODO - Finish docstring
    the loading, preprocessing and feature extraction for datasets.
    This class is used to manage datasets, including the

    Attributes
    ----------
    datasets : dict[str, DataFrame]
        A dictionary of dataset dataframes with the data set name as the key and the data
        set as loaded from pd.read_csv()
    preprocessor : Preprocessor
        Preprocessor to be used to preprocess data

    Methods
    -------
    method_name
        Description of the method.

    Examples
    --------
    Examples of how to use this class.
    """

    def __init__(self):
        self.datasets: dict[str, Dataset] = {}
        self.preprocessor: Preprocessor = Preprocessor()
        self.feature_extractor: FeatureExtractor = FeatureExtractor()
        logging.info(f"Dataset manager initialized.")

    def load_dataset(self, name: str, data_path: str) -> None:
        """
        loads a dataset into the datasets dictionary.
        # TODO finish docstring
        Args:
            name (string): name to be used for the key in the key value pair {'name',dataset}
            data (string): the dataframe

        Returns:
            None
        """

        dataset = Dataset()
        dataset.load_dataframe(data_path)
        self.set_dataset(dataset, name)

        logging.info(f"Dataset loaded as {name}.")

    def get_dataset(self, name: str) -> Dataset:
        try:
            return self.datasets[name]
        except KeyError:
            logging.error(f"Dataset '{name}' does not exist.")
            return None

    def set_dataset(self, dataset, name):
        """
        # TODO finish docstring
        set a dataset in the dictionary

        Args:
            param1 (type): Description of param1
            param2 (type): Description of param2

        Returns:
            type: Description of return value
        """

        self.datasets[name] = dataset

    def preprocess(self, name):
        # TODO: Implement preprocessing logic
        pass

    def feature_extraction(self, name):
        # TODO: Implement feature extraction logic
        pass
