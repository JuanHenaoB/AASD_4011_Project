# app.py
import logging
from src.logging_config import setup_logging
from src.PipelineManager import PipelineManager

# Set up logging for the application
setup_logging()

# Main entry point of the application
if __name__ == '__main__':
    # Initialize the PipelineManager
    pipeline_manager = PipelineManager()
    
    # Start the data processing pipeline
    # TODO: Add the specific data or parameters required to run the pipeline
    pipeline_manager.run_pipeline()

    logging.info("Application started successfully.")
