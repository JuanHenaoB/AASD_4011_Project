# app.py
import logging
from src.DatasetManager import DatasetManager
from src import logging_config

from src.logging_config import setup_logging
from src.PipelineManager import PipelineManager

# Set up logging for the application
setup_logging()

# Main entry point of the application
if __name__ == "__main__":
    # Initialize the PipelineManager
    pipeline_manager = PipelineManager()

    # Initialize the DatasetManager
    dataset_manager: DatasetManager = DatasetManager()
    dataset_manager.create_dataset("main", "all-data.csv")
    ds = dataset_manager.get_dataset("main")
    df = ds.dataframe
    print(df.head())
    # Start the data processing pipeline

    logging.info("Application started successfully.")
