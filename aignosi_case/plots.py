from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import typer

from aignosi_case.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> Path:
    """
    Save a matplotlib figure to the reports/figures directory.
    
    Args:
        fig: Matplotlib figure object to save
        filename: Name of the file (e.g., 'correlation_matrix.png')
        dpi: Resolution for the saved figure (default: 300)
        bbox_inches: Bounding box setting (default: 'tight')
    
    Returns:
        Path: Full path to the saved figure
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    logger.success(f"Saved figure to {output_path}")
    plt.close(fig)
    
    return output_path


def save_current_figure(filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> Path:
    """
    Save the current matplotlib figure (plt.gcf()) to the reports/figures directory.
    
    Args:
        filename: Name of the file (e.g., 'correlation_matrix.png')
        dpi: Resolution for the saved figure (default: 300)
        bbox_inches: Bounding box setting (default: 'tight')
    
    Returns:
        Path: Full path to the saved figure
    """
    fig = plt.gcf()
    return save_figure(fig, filename, dpi, bbox_inches)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
