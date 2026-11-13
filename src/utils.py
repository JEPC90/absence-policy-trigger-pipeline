import logging 
import os
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a standard professional logger that writes severity levels, timestamps, and messages
    to both the console and the daily log file.
    """
    
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Create a unique log file name based on the current date and time
    log_filename = os.path.join("logs", f"pipline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    # Create logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO) 
    
    #Prevent duplicate log entries if the logger is called multiple times
    if not logger.handlers:
        # File handler for writing logs to a file
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO) 
        
        # Console handler for outputting logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Define a standard log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        file_handler.setFormatter(formatter)   
        console_handler.setFormatter(formatter)
        
        #Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
