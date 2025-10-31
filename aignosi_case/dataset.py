from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from aignosi_case.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, get_csv_path

app = typer.Typer()


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw dataset.
    
    Returns:
        pd.DataFrame: Raw data with datetime index
    """
    logger.info(f"Loading raw data from {get_csv_path()}")
    df = pd.read_csv(get_csv_path(), parse_dates=[0], index_col=0, decimal=',')
    logger.success(f"Loaded {len(df)} rows of raw data")
    return df


def load_hourly_data() -> pd.DataFrame:
    """
    Load and resample the raw dataset to hourly frequency.
    
    Returns:
        pd.DataFrame: Hourly resampled data (mean aggregation)
    """
    df = load_raw_data()
    logger.info("Resampling to hourly frequency...")
    df_hourly = df.resample('h').mean()
    logger.success(f"Resampled to {len(df_hourly)} hourly data points")
    return df_hourly


def save_interim_data(df: pd.DataFrame, filename: str) -> Path:
    """
    Save a processed DataFrame to the interim data directory.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (should end in .csv, .parquet, etc.)
    
    Returns:
        Path: Full path to the saved file
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DATA_DIR / filename
    
    if filename.endswith('.parquet'):
        df.to_parquet(output_path)
    elif filename.endswith('.csv'):
        df.to_csv(output_path)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    logger.success(f"Saved interim data to {output_path}")
    return output_path


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
