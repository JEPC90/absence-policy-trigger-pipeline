import os
import pandas as pd
from src.utils import setup_logger

# Initialise the logger
logger = setup_logger(__name__)

def ingest_csv_to_bronze(file_path: str) -> pd.DataFrame:
    """
    Safely ingests a raw CSV file into the Bronze layer, performing existence checks and logging the process.
    """
    logger.info(f"Target file verification started: {file_path}")
    
    # Defensive check: Ensure the file exists before attempting to read it
    if not os.path.exists(file_path):
        logger.error(f"Ingestion failed: File not found at {file_path}")
        raise FileNotFoundError(f"Missing required raw data file: {file_path}")
    
    try:
        logger.info(f"Reading raw CSV into DataFrame: {file_path}")
        df = pd.read_csv(file_path)
        
        # Capture metadata
        row_count, col_count = df.shape
        logger.info(f"Successfully ingested {os.path.basename(file_path)} | Rows: {row_count:,} | Columns: {col_count:}")
        
        return df
    
    except Exception as e:
        logger.error(f"An unexpected error occurred whilst ingesting {file_path}: {e}")
        raise

def main():
    logger.info("===============================================")
    logger.info("Starting the Bronze Layer Ingestion Process")
    logger.info("===============================================")
    
    # Define the path to the raw CSV file
    bronze_dir = os.path.join("data", "bronze")
    attendance_file = os.path.join(bronze_dir, "attendance_raw_data.csv")
    employee_file = os.path.join(bronze_dir, "employee_raw_data.csv")
    
    # Ensure the target bronze directory exists
    os.makedirs(bronze_dir, exist_ok=True)
    
    try:
        # Ingest both raw tables
        attendance_df = ingest_csv_to_bronze(attendance_file)
        employee_df = ingest_csv_to_bronze(employee_file)
        
        logger.info("Bronze Layer Ingestion Process completed successfully.")
        logger.info("===============================================")
        
    except Exception as e:
        logger.critical(f"Bronze Layer Ingestion Process failed: {e}")
        logger.info("===============================================")
        raise

if __name__ == "__main__":
    main()
    
