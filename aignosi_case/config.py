"""
Configuration module for the aignosi_case project.

This module centralizes:
- Project paths (data, models, reports)
- Dataset configuration (CSV filename, helper to load)
- Plotting defaults (seaborn style, figure size)

Usage in notebooks:
    from aignosi_case.config import get_csv_path, PLOT_FIGSIZE, SEABORN_STYLE
    import pandas as pd

    df = pd.read_csv(get_csv_path(), parse_dates=[0], index_col=0, decimal=',')
"""
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset configuration
CSV_FILENAME = "MiningProcess_Flotation_Plant_Database.csv"

# Plotting defaults
PLOT_FIGSIZE = (12, 6)
SEABORN_STYLE = "whitegrid"


def get_csv_path() -> Path:
    """
    Returns the full path to the main CSV dataset.

    Returns:
        Path: Absolute path to MiningProcess_Flotation_Plant_Database.csv

    Raises:
        FileNotFoundError: If the CSV file does not exist in the expected location.
    """
    csv_path = RAW_DATA_DIR / CSV_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. "
            f"Please ensure {CSV_FILENAME} is in {RAW_DATA_DIR}."
        )
    return csv_path

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
