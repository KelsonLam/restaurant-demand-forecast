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
# Upscale steakhouse profile (Chop House style), Knoxville, TN.
LATITUDE = 35.9606
LONGITUDE = -83.9207
TIMEZONE = "America/New_York"

# US state for the holidays calendar.
HOLIDAY_STATE = "TN"
