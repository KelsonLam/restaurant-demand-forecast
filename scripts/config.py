"""Project-wide settings.

EDIT THESE to match the restaurant before running the pipeline.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Restaurant location, used to pull historical weather from Open-Meteo.
# Asian Eatery, Germantown, TN (Memphis metro, Central Time).
LATITUDE = 35.0868
LONGITUDE = -89.8101
TIMEZONE = "America/Chicago"

# US state for the holidays calendar.
HOLIDAY_STATE = "TN"
