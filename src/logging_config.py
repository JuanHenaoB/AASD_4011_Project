import logging
def setup_logging(level=logging.INFO):
    """
    Sets up basic logging configuration.
    
    Parameters:
    - level: Logging level, e.g., logging.INFO.
    """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    logging.info("Logging setup complete.")