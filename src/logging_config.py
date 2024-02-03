# logging_config.py
import logging

def setup_logging(level=logging.INFO, log_file='./logs/app.log'):
    """
    Sets up logging to both the console and a file.

    Parameters:
    - level: Logging level, e.g., logging.INFO.
    - log_file: The file to which logs will be written.
    """
    
    """
    Sets up logging to both the console and a file.
    
    Args:
        level (enum): Logging level, e.g., logging.INFO.
        log_file (string): file path of log file
    
    Returns:
        None
    """
    
    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Create a console handler for outputting logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("Logging setup complete, output directed to both console and file.")
