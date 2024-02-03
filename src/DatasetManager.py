from src.Preprocessor import Preprocessor
from src.FeatureExtractor import FeatureExtractor
import logging
from src.Dataset import Dataset


class DatasetManager:
    """
    Manages the loading, preprocessing, and feature extraction for datasets.

    This class is used to manage datasets, including the loading, preprocessing,
    and feature extraction operations. It provides methods to create and retrieve
    datasets, as well as access the preprocessor and feature extractor.

    Attributes
    ----------
    datasets : dict[str, Dataset]
        A dictionary of dataset objects with the dataset name as the key and the
        dataset as loaded from pd.read_csv().
    preprocessor : Preprocessor
        Preprocessor to be used to preprocess data.
    feature_extractor : FeatureExtractor
        Feature extractor to be used to extract features from data.

    Methods
    -------
    create_dataset(name: str, data_path: str) -> None:
        Creates and adds a dataset into the datasets dictionary.

    get_dataset(name: str) -> Dataset:
        Retrieves a dataset from the datasets dictionary.

    Examples
    --------
    Examples of how to use this class.
    """

    def __init__(self):
        self._datasets: dict[str, Dataset] = {}
        self._preprocessor: Preprocessor = Preprocessor()
        self._feature_extractor: FeatureExtractor = FeatureExtractor()
        logging.info(f"Dataset manager initialized.")

    @property
    def datasets(self) -> dict[str, Dataset]:
        return self._datasets

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @property
    def feature_extractor(self) -> FeatureExtractor:
        return self._feature_extractor

    def create_dataset(self, name: str, data_path: str) -> None:
        """
        Creates and adds a dataset into the datasets dictionary.

        Args:
            name (str): Name to be used as the key in the key-value pair {'name', dataset}.
            data_path (str): The path to the data file.

        Returns:
            None
        """

        dataset = Dataset()
        dataset.load_dataframe(data_path)
        self._datasets[name] = dataset
        logging.info(f"Dataset loaded as {name}.")

    def get_dataset(self, name: str) -> Dataset:
        """
        Retrieves a dataset from the datasets dictionary.

        Args:
            name (str): Name of the dataset to retrieve.

        Returns:
            Dataset: The dataset object if found, None otherwise.
        """
        return self._datasets.get(name, None)


