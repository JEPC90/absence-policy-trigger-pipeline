import sys
from src.utils import setup_logger
from src.transform_silver import main as run_silver_transformation
from src.analyse_gold import run_gold_layer_analysis

# Initialize orchestration logger
logger = setup_logger(__name__)

def main():
    """
    Orchestrates the full end-to-end absence policy pipeline:
    1. Bronze to Silver: Cleans, unifies, and merges attendance & employee master data.
    2. Silver to Gold: Executes Business-Day Gaps & Islands analysis and flags policy breaches.
    """
    logger.info("==========================================================================")
    logger.info("STARTING FULL ABSENCE POLICY PIPELINE EXECUTION")
    logger.info("==========================================================================")
    
    try:
        # Step 1: Execute Silver Transformation
        logger.info("Phase 1/2: Running Silver Layer Transformation...")
        run_silver_transformation()
        
        # Step 2: Execute Gold Analysis
        logger.info("Phase 2/2: Running Gold Layer Policy Breach Analysis...")
        run_gold_layer_analysis()
        
        logger.info("==========================================================================")
        logger.info("PIPELINE EXECUTION COMPLETE: Gold breach report successfully updated.")
        logger.info("==========================================================================")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed due to an unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()