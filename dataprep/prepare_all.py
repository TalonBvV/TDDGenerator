"""
Dataset Preparation - Main Entry Point

Complete pipeline for downloading MMOCR text detection datasets and
converting them to YOLOv8-Seg format for the TDDGenerator engine.

Usage:
    # Download and convert all standard datasets
    python -m dataprep.prepare_all --output-dir ./datasets
    
    # Download specific datasets
    python -m dataprep.prepare_all --datasets icdar2015 totaltext --output-dir ./datasets
    
    # Convert already downloaded data (skip download)
    python -m dataprep.prepare_all --skip-download --output-dir ./datasets
    
    # Include large datasets (synthtext ~40GB, cocotextv2 ~13GB)
    python -m dataprep.prepare_all --include-large --output-dir ./datasets
"""

import argparse
import logging
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from tqdm import tqdm

from .download_datasets import MMOCRDownloader, TEXTDET_DATASETS
from .convert_to_yolo import MMOCRToYOLOConverter, ConversionStats

logger = logging.getLogger("dataprep.main")


@dataclass
class PrepareConfig:
    """Configuration for dataset preparation."""
    # Datasets to process
    datasets: List[str] = None
    
    # Directories
    output_dir: Path = Path("./datasets")
    mmocr_data_dir: Path = Path("./data")
    
    # Options
    workers: int = 4
    skip_download: bool = False
    skip_convert: bool = False
    include_large: bool = False
    force_download: bool = False
    copy_images: bool = True
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = self._get_default_datasets()
        self.output_dir = Path(self.output_dir)
        self.mmocr_data_dir = Path(self.mmocr_data_dir)
    
    def _get_default_datasets(self) -> List[str]:
        """Get default (non-large) datasets."""
        return [
            name for name, config in TEXTDET_DATASETS.items()
            if config.size_estimate_gb <= 10
        ]
    
    @classmethod
    def from_yaml(cls, path: Path) -> "PrepareConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Handle nested structure
        config_dict = {}
        
        if "datasets" in data:
            config_dict["datasets"] = data["datasets"]
        if "output_dir" in data:
            config_dict["output_dir"] = data["output_dir"]
        if "mmocr_data_dir" in data:
            config_dict["mmocr_data_dir"] = data["mmocr_data_dir"]
        if "workers" in data:
            config_dict["workers"] = data["workers"]
        if "options" in data:
            options = data["options"]
            config_dict.update({
                "skip_download": options.get("skip_download", False),
                "skip_convert": options.get("skip_convert", False),
                "include_large": options.get("include_large", False),
                "force_download": options.get("force_download", False),
                "copy_images": options.get("copy_images", True),
            })
        
        return cls(**config_dict)


@dataclass
class PrepareResult:
    """Results from dataset preparation."""
    timestamp: str
    config: Dict[str, Any]
    download_results: Dict[str, bool]
    conversion_results: Dict[str, Dict[str, Any]]
    total_images: int = 0
    total_polygons: int = 0
    success: bool = True


class DatasetPreparer:
    """Main orchestrator for dataset preparation."""
    
    def __init__(self, config: PrepareConfig):
        self.config = config
        
        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.mmocr_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize downloader
        self.downloader = MMOCRDownloader(
            data_dir=self.config.mmocr_data_dir,
        )
    
    def prepare(self) -> PrepareResult:
        """
        Run the full preparation pipeline.
        
        Returns:
            PrepareResult with all results and statistics
        """
        result = PrepareResult(
            timestamp=datetime.now().isoformat(),
            config=asdict(self.config),
            download_results={},
            conversion_results={},
        )
        
        # Determine datasets to process
        datasets = self.config.datasets
        if "all" in datasets:
            datasets = list(TEXTDET_DATASETS.keys())
            if not self.config.include_large:
                datasets = [
                    d for d in datasets 
                    if TEXTDET_DATASETS[d].size_estimate_gb <= 10
                ]
        
        logger.info(f"Preparing {len(datasets)} dataset(s): {', '.join(datasets)}")
        
        # Step 1: Download
        if not self.config.skip_download:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 1: Downloading datasets")
            logger.info("=" * 60)
            
            result.download_results = self.downloader.download_datasets(
                dataset_names=datasets,
                force=self.config.force_download,
                workers=1,  # Sequential downloads are more reliable
                skip_large=not self.config.include_large,
            )
        else:
            logger.info("Skipping download step")
            result.download_results = {d: True for d in datasets}
        
        # Step 2: Convert
        if not self.config.skip_convert:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: Converting to YOLOv8-Seg format")
            logger.info("=" * 60)
            
            for dataset_name in tqdm(datasets, desc="Converting"):
                # Check if dataset was downloaded/exists
                dataset_dir = self.config.mmocr_data_dir / dataset_name
                
                if not dataset_dir.exists():
                    logger.warning(f"Dataset directory not found: {dataset_dir}")
                    result.conversion_results[dataset_name] = {
                        "success": False,
                        "error": "Directory not found",
                    }
                    continue
                
                # Output directory for this dataset
                output_dir = self.config.output_dir / dataset_name
                
                try:
                    converter = MMOCRToYOLOConverter(
                        input_dir=dataset_dir,
                        output_dir=output_dir,
                        copy_images=self.config.copy_images,
                        workers=self.config.workers,
                    )
                    
                    stats = converter.convert()
                    
                    result.conversion_results[dataset_name] = {
                        "success": stats.images_processed > 0,
                        "images_processed": stats.images_processed,
                        "images_skipped": stats.images_skipped,
                        "polygons_converted": stats.polygons_converted,
                        "polygons_ignored": stats.polygons_ignored,
                        "errors": len(stats.errors),
                    }
                    
                    result.total_images += stats.images_processed
                    result.total_polygons += stats.polygons_converted
                    
                except Exception as e:
                    logger.error(f"Failed to convert {dataset_name}: {e}")
                    result.conversion_results[dataset_name] = {
                        "success": False,
                        "error": str(e),
                    }
        else:
            logger.info("Skipping conversion step")
        
        # Determine overall success
        result.success = (
            all(result.download_results.values()) and
            all(r.get("success", False) for r in result.conversion_results.values())
        )
        
        return result
    
    def save_report(self, result: PrepareResult, output_path: Optional[Path] = None) -> Path:
        """Save preparation report to JSON file."""
        if output_path is None:
            output_path = self.config.output_dir / "prepare_report.json"
        
        # Convert result to serializable format
        report = {
            "timestamp": result.timestamp,
            "success": result.success,
            "total_images": result.total_images,
            "total_polygons": result.total_polygons,
            "config": {
                "datasets": list(result.config.get("datasets", [])),
                "output_dir": str(result.config.get("output_dir", "")),
                "mmocr_data_dir": str(result.config.get("mmocr_data_dir", "")),
            },
            "download_results": result.download_results,
            "conversion_results": result.conversion_results,
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {output_path}")
        return output_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and convert MMOCR datasets to YOLOv8-Seg format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and convert standard datasets
  python -m dataprep.prepare_all --output-dir ./datasets
  
  # Specific datasets only
  python -m dataprep.prepare_all --datasets icdar2015 totaltext --output-dir ./datasets
  
  # Skip download, just convert existing data
  python -m dataprep.prepare_all --skip-download --output-dir ./datasets
  
  # Include large datasets (synthtext, cocotextv2, textocr)
  python -m dataprep.prepare_all --include-large --output-dir ./datasets
  
  # Use config file
  python -m dataprep.prepare_all --config dataprep/config.yaml
        """
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["all"],
        help="Datasets to download/convert, or 'all' for all (default: all non-large)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./datasets"),
        help="Output directory for converted datasets (default: ./datasets)"
    )
    parser.add_argument(
        "--mmocr-data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory for MMOCR raw data (default: ./data)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file (YAML)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers for conversion (default: 4)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing data)"
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion step (download only)"
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include large datasets (>10GB): synthtext, cocotextv2, textocr"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if datasets exist"
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Don't copy images (create labels only)"
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
    
    return parser.parse_args()


def print_summary(result: PrepareResult):
    """Print a summary of the preparation results."""
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    
    print(f"\nTimestamp: {result.timestamp}")
    print(f"Success: {'✓ Yes' if result.success else '✗ No'}")
    
    print(f"\nDownloaded datasets: {sum(result.download_results.values())}/{len(result.download_results)}")
    for name, success in sorted(result.download_results.items()):
        status = "✓" if success else "✗"
        print(f"  [{status}] {name}")
    
    print(f"\nConverted datasets:")
    for name, stats in sorted(result.conversion_results.items()):
        success = stats.get("success", False)
        status = "✓" if success else "✗"
        images = stats.get("images_processed", 0)
        polygons = stats.get("polygons_converted", 0)
        print(f"  [{status}] {name}: {images} images, {polygons} polygons")
    
    print(f"\nTotals:")
    print(f"  Images:   {result.total_images:,}")
    print(f"  Polygons: {result.total_polygons:,}")
    print()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # List mode
    if args.list:
        print("\nAvailable text detection datasets:")
        print("-" * 60)
        for name, config in sorted(TEXTDET_DATASETS.items()):
            size = f"~{config.size_estimate_gb:.1f} GB"
            large = " [LARGE]" if config.size_estimate_gb > 10 else ""
            print(f"  {name:20} {size:>10}{large}")
        print()
        return 0
    
    # Load config
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = PrepareConfig.from_yaml(args.config)
        # Override with CLI args
        if args.datasets != ["all"]:
            config.datasets = args.datasets
        config.output_dir = args.output_dir
        config.mmocr_data_dir = args.mmocr_data_dir
        config.workers = args.workers
        config.skip_download = args.skip_download or config.skip_download
        config.skip_convert = args.skip_convert or config.skip_convert
        config.include_large = args.include_large or config.include_large
        config.force_download = args.force or config.force_download
        config.copy_images = not args.no_copy_images
    else:
        config = PrepareConfig(
            datasets=args.datasets,
            output_dir=args.output_dir,
            mmocr_data_dir=args.mmocr_data_dir,
            workers=args.workers,
            skip_download=args.skip_download,
            skip_convert=args.skip_convert,
            include_large=args.include_large,
            force_download=args.force,
            copy_images=not args.no_copy_images,
        )
    
    # Run preparation
    preparer = DatasetPreparer(config)
    result = preparer.prepare()
    
    # Save report
    preparer.save_report(result)
    
    # Print summary
    print_summary(result)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
