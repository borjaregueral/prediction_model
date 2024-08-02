from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
BRONZE_DATA_DIR = DATA_DIR / 'bronze'
SILVER_DATA_DIR = DATA_DIR / 'silver'
GOLD_DATA_DIR = DATA_DIR / 'gold'

def create_directories():
    """
    Creates the necessary directories if they do not exist.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f'Folder "data" ensured at "{DATA_DIR}"')
        
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f'Folder "raw" ensured at "{RAW_DATA_DIR}"')
        
        BRONZE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f'Folder "processed" ensured at "{BRONZE_DATA_DIR}"')
        
        SILVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f'Folder "transformed" ensured at "{SILVER_DATA_DIR}"')
        
        GOLD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f'Folder "transformed" ensured at "{GOLD_DATA_DIR}"')
        
    except Exception as e:
        
        logging.error(f"An error occurred while creating directories: {e}")

# Call the function to create directories when the module is run directly
if __name__ == "__main__":
    create_directories()