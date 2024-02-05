# app.py
import logging
from src.DataPipeline import DataPipeline
from src.DatasetManager import DatasetManager
from src.logging_config import setup_logging
from src.PipelineManager import PipelineManager
from src.PreprocessorStrategies import (
    BalancingOneHot,
    OneHotEncoding,
    Tokenization,
    Balancing,
)
from src.FeatureExtractionStrategies import TFIDFExtraction
from src import logging_config

from src.DataPipeline import DataPipeline
from src.DatasetManager import DatasetManager
from src.PipelineManager import PipelineManager
from src.PreprocessorStrategies import Tokenization, Balancing
from src.FeatureExtractionStrategies import TFIDFExtraction
from src.PreprocessorStrategies import CombineWithOriginalData


"""
    Entry point for the application
    
    I regret everything...


    Overview
    ----------
    DatasetManager is responsible for loading, storing, and accessing datasets.
    PipelineManager could be responsible for managing multiple DataPipeline instances, each configured for different preprocessing and feature extraction tasks.
    DataPipeline represents a specific sequence of data processing steps (e.g., cleaning, tokenization, vectorization).

    Methods
    -------
    TODO: proper documentation

    Examples
    --------
    Just use it.
"""

# Set up logging for the application
setup_logging()

if __name__ == "__main__":
    # Initialize the DatasetManager and load the dataset
    dataset_manager = DatasetManager()
    dataset_manager.load_dataframe("main", "all-data.csv")
    df = dataset_manager.active_dataframe
    original_df = df.copy()

    # Check if the DataFrame is loaded correctly
    print(df.head())

    # Initialize the PipelineManager
    pipeline_manager = PipelineManager()

    # Create the DataPipeline with preprocessors and feature extractors
    tokenizer = Tokenization()
    balancer = Balancing()
    tfidf_extractor = TFIDFExtraction()
    pipeline = DataPipeline(
        preprocessors=[tokenizer, balancer], feature_extractors=[tfidf_extractor]
    )

    # Add the pipeline to the PipelineManager
    pipeline_manager.add_pipeline("text_processing", pipeline)

    if "label" in df.columns:
        processed_data = pipeline_manager.run_pipeline("text_processing", df[["label"]])
        print(processed_data)
    else:
        logging.error("Column 'text' not found in the DataFrame.")
    # Initialize the preprocessors
    print(processed_data)

    label_column = "label"
    one_hot_encoder = OneHotEncoding(label_column=label_column)
    balancing_one_hot = BalancingOneHot()

    # Create the pipeline with one-hot encoding followed by balancing
    pipeline = DataPipeline(
        preprocessors=[
            one_hot_encoder,
            balancing_one_hot,
            CombineWithOriginalData(original_df),
        ]
    )
    pipeline2 = DataPipeline(
        preprocessors=[one_hot_encoder, CombineWithOriginalData(original_df)]
    )

    pipeline_manager.add_pipeline("onehot_balanced", pipeline)
    pipeline_manager.add_pipeline("onehot_encoded", pipeline2)

    processed_data = pipeline_manager.run_pipeline("onehot_balanced", df[["label"]])
    processed_data2 = pipeline_manager.run_pipeline("onehot_encoded", df[["label"]])

    print(processed_data)
