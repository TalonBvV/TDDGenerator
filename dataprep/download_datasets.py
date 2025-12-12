"""
Standalone Dataset Downloader

Downloads text detection datasets without any MM* dependencies.
Uses requests for HTTP downloads and standard library for archive extraction.
"""

import hashlib
import logging
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from .dataset_configs import (
    DatasetConfig,
    FileConfig,
    SplitConfig,
    get_available_datasets,
    get_dataset_config,
)

logger = logging.getLogger("dataprep.download")


class DatasetDownloader:
    """Standalone dataset downloader with no MM* dependencies."""

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize downloader.

        Args:
            data_dir: Directory where datasets will be stored
            cache_dir: Cache directory for downloads (default: data_dir/cache)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")

    def download_dataset(
        self,
        dataset_name: str,
        splits: List[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Download a single dataset.

        Args:
            dataset_name: Name of the dataset (e.g., "icdar2015")
            splits: Splits to download (default: ["train", "test"])
            force: Force re-download even if exists

        Returns:
            True if successful
        """
        config = get_dataset_config(dataset_name)
        if config is None:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available: {', '.join(get_available_datasets())}")
            return False

        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if splits is None:
            splits = ["train", "test"]

        success = True
        for split in splits:
            split_config = getattr(config, split, None)
            if split_config is None:
                logger.warning(f"No {split} split for {dataset_name}")
                continue

            logger.info(f"Downloading {dataset_name} {split} split...")
            if not self._download_split(dataset_dir, split_config, force):
                success = False

        return success

    def _download_split(
        self,
        dataset_dir: Path,
        split_config: SplitConfig,
        force: bool,
    ) -> bool:
        """Download files for a dataset split."""
        for file_cfg in split_config.files:
            # Download file
            download_path = self.cache_dir / file_cfg.save_name

            if force or not self._check_integrity(download_path, file_cfg.md5):
                if not self._download_file(file_cfg.url, download_path):
                    return False

            # Extract archive
            if not self._extract_archive(download_path, dataset_dir):
                return False

            # Apply file mappings
            for src, dst in file_cfg.mapping:
                self._move_files(dataset_dir, src, dst)

        return True

    def _download_file(self, url: str, dst_path: Path) -> bool:
        """Download a file with progress bar."""
        logger.info(f"Downloading: {url}")
        logger.info(f"To: {dst_path}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(dst_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=dst_path.name,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            return True

        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            return False

    def _check_integrity(self, path: Path, expected_md5: Optional[str]) -> bool:
        """Check file integrity using MD5."""
        if not path.exists():
            return False

        if expected_md5 is None:
            return True  # No verification needed

        with open(path, "rb") as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()

        return file_md5 == expected_md5

    def _extract_archive(self, src_path: Path, dst_dir: Path) -> bool:
        """Extract zip or tar.gz archive."""
        if not src_path.exists():
            return False

        name = src_path.name
        
        # Skip if not an archive
        if not (name.endswith(".zip") or name.endswith(".tar.gz") or name.endswith(".tar")):
            return True

        # Check if already extracted
        extract_marker = dst_dir / f".{src_path.stem}.extracted"
        if extract_marker.exists():
            logger.info(f"Already extracted: {name}")
            return True

        logger.info(f"Extracting: {name}")

        try:
            if name.endswith(".zip"):
                with zipfile.ZipFile(src_path, "r") as zf:
                    zf.extractall(dst_dir)
            elif name.endswith(".tar.gz"):
                with tarfile.open(src_path, "r:gz") as tf:
                    tf.extractall(dst_dir)
            elif name.endswith(".tar"):
                with tarfile.open(src_path, "r:") as tf:
                    tf.extractall(dst_dir)

            # Mark as extracted
            extract_marker.touch()
            return True

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False

    def _move_files(self, base_dir: Path, src: str, dst: str) -> None:
        """Move/rename files within dataset directory."""
        src_path = base_dir / src
        dst_path = base_dir / dst

        if not src_path.exists():
            return

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            return  # Already moved

        try:
            shutil.move(str(src_path), str(dst_path))
            logger.debug(f"Moved: {src} -> {dst}")
        except Exception as e:
            logger.warning(f"Failed to move {src}: {e}")

    def download_datasets(
        self,
        dataset_names: List[str],
        splits: List[str] = None,
        force: bool = False,
        skip_large: bool = True,
    ) -> Dict[str, bool]:
        """
        Download multiple datasets.

        Args:
            dataset_names: List of dataset names
            splits: Splits to download
            force: Force re-download
            skip_large: Skip datasets > 10GB

        Returns:
            Dict mapping dataset names to success status
        """
        results = {}

        for name in tqdm(dataset_names, desc="Downloading datasets"):
            config = get_dataset_config(name)
            if config is None:
                logger.warning(f"Unknown dataset: {name}")
                results[name] = False
                continue

            if skip_large and config.size_estimate_gb > 10:
                logger.info(f"Skipping large dataset: {name} (~{config.size_estimate_gb:.1f} GB)")
                results[name] = False
                continue

            results[name] = self.download_dataset(name, splits, force)

        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Downloaded {success_count}/{len(results)} datasets successfully")

        return results


def get_image_annotation_pairs(
    dataset_dir: Path,
    split: str,
    split_config: SplitConfig,
) -> List[Tuple[Path, Path]]:
    """
    Get list of (image_path, annotation_path) pairs.

    Args:
        dataset_dir: Root directory of the dataset
        split: Split name (train/test)
        split_config: Configuration for this split

    Returns:
        List of (image_path, annotation_path) tuples
    """
    img_dir = dataset_dir / "images" / split
    ann_dir = dataset_dir / "annotations" / split

    if not img_dir.exists():
        logger.warning(f"Image directory not found: {img_dir}")
        return []

    pairs = []
    img_pattern = re.compile(split_config.img_pattern)

    for img_path in img_dir.iterdir():
        if not img_path.is_file():
            continue

        match = img_pattern.match(img_path.name)
        if not match:
            continue

        # Get annotation filename from pattern
        img_id = match.group(1)
        ann_name = split_config.ann_pattern.format(img_id)
        ann_path = ann_dir / ann_name

        if ann_path.exists():
            pairs.append((img_path, ann_path))

    return pairs


def main():
    """CLI entry point for dataset download."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download text detection datasets (no MM* dependencies)"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["all"],
        help="Dataset names to download, or 'all' for all datasets",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./data"),
        help="Output directory for datasets (default: ./data)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if exists",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include large datasets (>10GB)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.list:
        print("\nAvailable text detection datasets:")
        print("-" * 50)
        for name in get_available_datasets():
            config = get_dataset_config(name)
            size = f"~{config.size_estimate_gb:.1f} GB"
            print(f"  {name:15} {size:>10}")
        print()
        return 0

    downloader = DatasetDownloader(data_dir=args.output_dir)

    # Determine datasets to download
    if "all" in args.datasets:
        dataset_names = get_available_datasets()
    else:
        dataset_names = args.datasets

    results = downloader.download_datasets(
        dataset_names,
        force=args.force,
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
    import sys
    sys.exit(main())
