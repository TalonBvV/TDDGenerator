"""
Dataset preparation module for downloading text detection datasets
and converting to YOLOv8-Seg format.

NO MM* DEPENDENCIES - uses standalone download and parsing.

This module provides tools to:
1. Download text detection datasets (ICDAR, TotalText, CTW1500, etc.)
2. Parse various annotation formats
3. Convert to YOLOv8-Seg format for the TDDGenerator engine
"""

from .dataset_configs import (
    DATASETS,
    DatasetConfig,
    get_available_datasets,
    get_dataset_config,
)
from .download_datasets import DatasetDownloader

__version__ = "2.0.0"

__all__ = [
    "DatasetDownloader",
    "DatasetConfig",
    "DATASETS",
    "get_available_datasets",
    "get_dataset_config",
]
