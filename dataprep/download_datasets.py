"""
MMOCR Dataset Downloader

Downloads text detection datasets using MMOCR's dataset preparation tools.
Supports all MMOCR-compatible text detection datasets.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

logger = logging.getLogger("dataprep.download")


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    task: str = "textdet"
    splits: List[str] = field(default_factory=lambda: ["train", "test"])
    size_estimate_gb: float = 1.0
    requires_manual: bool = False
    manual_instructions: str = ""


# All supported text detection datasets from MMOCR
TEXTDET_DATASETS: Dict[str, DatasetConfig] = {
    # ICDAR Series - Competition datasets
    "icdar2013": DatasetConfig(
        name="icdar2013",
        size_estimate_gb=0.5,
        requires_manual=False,
    ),
    "icdar2015": DatasetConfig(
        name="icdar2015",
        size_estimate_gb=0.5,
        requires_manual=False,
    ),
    
    # Curved/Arbitrary text datasets
    "ctw1500": DatasetConfig(
        name="ctw1500",
        size_estimate_gb=0.8,
        requires_manual=False,
    ),
    "totaltext": DatasetConfig(
        name="totaltext",
        size_estimate_gb=0.5,
        requires_manual=False,
    ),
    
    # Large-scale datasets
    "cocotextv2": DatasetConfig(
        name="cocotextv2",
        size_estimate_gb=13.0,
        requires_manual=False,
    ),
    "synthtext": DatasetConfig(
        name="synthtext",
        size_estimate_gb=40.0,
        splits=["train"],  # SynthText has no test split
        requires_manual=False,
    ),
    
    # Receipt/Document datasets
    "sroie": DatasetConfig(
        name="sroie",
        size_estimate_gb=0.1,
        requires_manual=False,
    ),
    "funsd": DatasetConfig(
        name="funsd",
        size_estimate_gb=0.05,
        requires_manual=False,
    ),
    
    # Other datasets
    "naf": DatasetConfig(
        name="naf",
        size_estimate_gb=0.3,
        requires_manual=False,
    ),
    "svt": DatasetConfig(
        name="svt",
        size_estimate_gb=0.1,
        splits=["train"],  # SVT has only training set in preparer
        requires_manual=False,
    ),
    "textocr": DatasetConfig(
        name="textocr",
        size_estimate_gb=7.0,
        requires_manual=False,
    ),
    "wildreceipt": DatasetConfig(
        name="wildreceipt",
        size_estimate_gb=0.2,
        requires_manual=False,
    ),
}


class MMOCRDownloader:
    """Downloads datasets using MMOCR's dataset preparation tools."""
    
    def __init__(
        self,
        data_dir: Path,
        mmocr_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the MMOCR downloader.
        
        Args:
            data_dir: Directory where datasets will be downloaded
            mmocr_path: Path to MMOCR installation (auto-detected if not provided)
            cache_dir: Cache directory for downloads
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to find MMOCR installation
        self.mmocr_path = self._find_mmocr_path(mmocr_path)
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
        
    def _find_mmocr_path(self, provided_path: Optional[Path]) -> Optional[Path]:
        """Find MMOCR installation path."""
        if provided_path and provided_path.exists():
            return Path(provided_path)
        
        # Try to find via pip
        try:
            import mmocr
            return Path(mmocr.__file__).parent.parent
        except ImportError:
            logger.warning("MMOCR not found in Python environment")
            return None
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(TEXTDET_DATASETS.keys())
    
    def get_dataset_info(self, name: str) -> Optional[DatasetConfig]:
        """Get configuration for a specific dataset."""
        return TEXTDET_DATASETS.get(name.lower())
    
    def is_downloaded(self, dataset_name: str) -> bool:
        """Check if a dataset is already downloaded."""
        dataset_dir = self.data_dir / dataset_name
        
        if not dataset_dir.exists():
            return False
        
        # Check for expected files
        expected_files = [
            "textdet_train.json",
            "textdet_imgs/train",
        ]
        
        for file in expected_files:
            if not (dataset_dir / file).exists():
                return False
        
        return True
    
    def download_dataset(
        self,
        dataset_name: str,
        force: bool = False,
        splits: Optional[List[str]] = None,
    ) -> bool:
        """
        Download a single dataset.
        
        Args:
            dataset_name: Name of the dataset to download
            force: Force re-download even if exists
            splits: Specific splits to download (default: all)
            
        Returns:
            True if successful, False otherwise
        """
        dataset_name = dataset_name.lower()
        config = TEXTDET_DATASETS.get(dataset_name)
        
        if config is None:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available datasets: {', '.join(self.get_available_datasets())}")
            return False
        
        if config.requires_manual:
            logger.warning(f"Dataset '{dataset_name}' requires manual download:")
            logger.warning(config.manual_instructions)
            return False
        
        if not force and self.is_downloaded(dataset_name):
            logger.info(f"Dataset '{dataset_name}' already downloaded, skipping")
            return True
        
        logger.info(f"Downloading dataset: {dataset_name} (~{config.size_estimate_gb:.1f} GB)")
        
        # Build the command for MMOCR's prepare_dataset.py
        cmd = [
            sys.executable,
            "-m", "mmocr.datasets.preparers.prepare_dataset",
            dataset_name,
            "--task", config.task,
        ]
        
        if splits:
            cmd.extend(["--splits"] + splits)
        
        # Add data root
        env = os.environ.copy()
        env["MMOCR_DATA_ROOT"] = str(self.data_dir)
        
        try:
            # Change to the data directory
            result = subprocess.run(
                cmd,
                cwd=str(self.data_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600 * 4,  # 4 hour timeout for large datasets
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to download {dataset_name}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return False
            
            logger.info(f"Successfully downloaded: {dataset_name}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout downloading {dataset_name}")
            return False
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return False
    
    def download_datasets(
        self,
        dataset_names: List[str],
        force: bool = False,
        workers: int = 1,
        skip_large: bool = True,
    ) -> Dict[str, bool]:
        """
        Download multiple datasets.
        
        Args:
            dataset_names: List of dataset names to download
            force: Force re-download
            workers: Number of parallel downloads (not recommended > 2)
            skip_large: Skip datasets > 10GB
            
        Returns:
            Dict mapping dataset names to success status
        """
        results = {}
        
        # Filter datasets
        to_download = []
        for name in dataset_names:
            config = TEXTDET_DATASETS.get(name.lower())
            if config is None:
                logger.warning(f"Unknown dataset: {name}")
                results[name] = False
                continue
            
            if skip_large and config.size_estimate_gb > 10:
                logger.info(f"Skipping large dataset: {name} (~{config.size_estimate_gb:.1f} GB)")
                results[name] = False
                continue
            
            to_download.append(name)
        
        if workers == 1:
            # Sequential download
            for name in tqdm(to_download, desc="Downloading datasets"):
                results[name] = self.download_dataset(name, force=force)
        else:
            # Parallel download
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.download_dataset, name, force): name
                    for name in to_download
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        logger.error(f"Error downloading {name}: {e}")
                        results[name] = False
        
        # Summary
        success = sum(1 for v in results.values() if v)
        logger.info(f"Downloaded {success}/{len(results)} datasets successfully")
        
        return results
    
    def download_all(
        self,
        force: bool = False,
        skip_large: bool = True,
        workers: int = 1,
    ) -> Dict[str, bool]:
        """
        Download all supported datasets.
        
        Args:
            force: Force re-download
            skip_large: Skip datasets > 10GB (synthtext, cocotextv2, textocr)
            workers: Number of parallel downloads
            
        Returns:
            Dict mapping dataset names to success status
        """
        all_datasets = list(TEXTDET_DATASETS.keys())
        return self.download_datasets(all_datasets, force=force, workers=workers, skip_large=skip_large)


def main():
    """CLI entry point for dataset download."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download MMOCR text detection datasets"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["all"],
        help="Dataset names to download, or 'all' for all datasets"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./data"),
        help="Output directory for datasets (default: ./data)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if already exists"
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include large datasets (>10GB)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel downloads"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    downloader = MMOCRDownloader(data_dir=args.output_dir)
    
    if args.list:
        print("\nAvailable text detection datasets:")
        print("-" * 50)
        for name, config in sorted(TEXTDET_DATASETS.items()):
            status = "✓" if downloader.is_downloaded(name) else " "
            size = f"~{config.size_estimate_gb:.1f} GB"
            manual = " (manual)" if config.requires_manual else ""
            print(f"  [{status}] {name:15} {size:>10}{manual}")
        print()
        return 0
    
    # Determine which datasets to download
    if "all" in args.datasets:
        dataset_names = list(TEXTDET_DATASETS.keys())
    else:
        dataset_names = args.datasets
    
    # Download
    results = downloader.download_datasets(
        dataset_names,
        force=args.force,
        workers=args.workers,
        skip_large=not args.include_large,
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Download Summary:")
    print("=" * 50)
    for name, success in sorted(results.items()):
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name:20} {status}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
