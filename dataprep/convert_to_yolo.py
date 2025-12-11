"""
MMOCR to YOLOv8-Seg Format Converter

Converts MMOCR JSON annotation format to YOLOv8-Seg format for the TDDGenerator engine.

MMOCR Format (JSON):
{
    "metainfo": {...},
    "data_list": [
        {
            "img_path": "path/to/image.jpg",
            "height": 480,
            "width": 640,
            "instances": [
                {
                    "polygon": [x1, y1, x2, y2, ...],  # pixel coordinates
                    "ignore": false
                }
            ]
        }
    ]
}

YOLOv8-Seg Format (TXT per image):
<class_id> x1 y1 x2 y2 ... xn yn
(coordinates normalized to 0-1)
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from PIL import Image

logger = logging.getLogger("dataprep.convert")


@dataclass
class ConversionStats:
    """Statistics from a conversion run."""
    images_processed: int = 0
    images_skipped: int = 0
    polygons_converted: int = 0
    polygons_ignored: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def load_mmocr_json(json_path: Path) -> Dict[str, Any]:
    """
    Load and parse MMOCR annotation JSON file.
    
    Args:
        json_path: Path to the JSON annotation file
        
    Returns:
        Parsed JSON data as dictionary
    """
    try:
        import orjson
        with open(json_path, "rb") as f:
            return orjson.loads(f.read())
    except ImportError:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)


def normalize_polygon(
    polygon: List[float],
    width: int,
    height: int,
    clamp: bool = True,
) -> List[Tuple[float, float]]:
    """
    Normalize polygon coordinates from pixels to 0-1 range.
    
    Args:
        polygon: List of pixel coordinates [x1, y1, x2, y2, ...]
        width: Image width in pixels
        height: Image height in pixels
        clamp: Whether to clamp values to [0, 1]
        
    Returns:
        List of normalized (x, y) tuples
    """
    if len(polygon) % 2 != 0:
        raise ValueError(f"Polygon has odd number of coordinates: {len(polygon)}")
    
    points = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / width
        y = polygon[i + 1] / height
        
        if clamp:
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
        
        points.append((x, y))
    
    return points


def write_yolo_label(
    polygons: List[List[Tuple[float, float]]],
    output_path: Path,
    class_id: int = 0,
) -> None:
    """
    Write polygons to a YOLOv8-Seg format label file.
    
    Args:
        polygons: List of polygons, each as list of (x, y) normalized points
        output_path: Path to output .txt file
        class_id: Class ID (default 0 for text)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    for points in polygons:
        if len(points) < 3:
            continue  # Skip degenerate polygons
        
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        lines.append(f"{class_id} {coords}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def convert_single_image(
    image_info: Dict[str, Any],
    input_image_dir: Path,
    output_image_dir: Path,
    output_label_dir: Path,
    copy_images: bool = True,
) -> Tuple[bool, int, int]:
    """
    Convert a single image's annotations from MMOCR to YOLO format.
    
    Args:
        image_info: Image info dict from MMOCR data_list
        input_image_dir: Source directory for images
        output_image_dir: Destination directory for images
        output_label_dir: Destination directory for labels
        copy_images: Whether to copy images to output directory
        
    Returns:
        Tuple of (success, num_polygons, num_ignored)
    """
    img_path = image_info.get("img_path", "")
    width = image_info.get("width", 0)
    height = image_info.get("height", 0)
    instances = image_info.get("instances", [])
    
    if not img_path or not width or not height:
        return False, 0, 0
    
    # Handle relative/absolute paths
    img_name = Path(img_path).name
    src_img_path = input_image_dir / img_name
    
    # If image doesn't exist directly, try to find it
    if not src_img_path.exists():
        # Try the full path as provided
        alt_path = input_image_dir.parent / img_path
        if alt_path.exists():
            src_img_path = alt_path
        else:
            return False, 0, 0
    
    # Convert polygons
    polygons = []
    ignored = 0
    
    for instance in instances:
        # Skip ignored instances
        if instance.get("ignore", False):
            ignored += 1
            continue
        
        polygon = instance.get("polygon", [])
        if not polygon or len(polygon) < 6:  # Need at least 3 points
            ignored += 1
            continue
        
        try:
            normalized = normalize_polygon(polygon, width, height)
            if len(normalized) >= 3:
                polygons.append(normalized)
        except Exception as e:
            logger.debug(f"Error normalizing polygon: {e}")
            ignored += 1
    
    if not polygons:
        return False, 0, ignored
    
    # Write label file
    stem = src_img_path.stem
    label_path = output_label_dir / f"{stem}.txt"
    write_yolo_label(polygons, label_path)
    
    # Copy image if requested
    if copy_images:
        dst_img_path = output_image_dir / img_name
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
    
    return True, len(polygons), ignored


class MMOCRToYOLOConverter:
    """Converts MMOCR dataset format to YOLOv8-Seg format."""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        copy_images: bool = True,
        workers: int = 4,
    ):
        """
        Initialize the converter.
        
        Args:
            input_dir: MMOCR dataset directory (e.g., data/icdar2015)
            output_dir: Output directory for YOLOv8 format
            copy_images: Whether to copy images to output
            workers: Number of parallel workers
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.copy_images = copy_images
        self.workers = workers
    
    def find_annotation_files(self) -> List[Path]:
        """Find all MMOCR annotation JSON files."""
        patterns = [
            "textdet_train.json",
            "textdet_test.json",
            "textdet_val.json",
        ]
        
        found = []
        for pattern in patterns:
            files = list(self.input_dir.glob(pattern))
            found.extend(files)
        
        return found
    
    def detect_image_directory(self, json_path: Path) -> Path:
        """Detect the image directory for a given annotation file."""
        # Try common locations
        base = json_path.parent
        
        # Parse split from filename (textdet_train.json -> train)
        stem = json_path.stem
        split = stem.replace("textdet_", "")  # train, test, val
        
        candidates = [
            base / "textdet_imgs" / split,
            base / "textdet_imgs",
            base / f"{split}",
            base / "images" / split,
            base / "images",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # Fallback to parent
        return base
    
    def convert_split(
        self,
        json_path: Path,
        split_name: str,
    ) -> ConversionStats:
        """
        Convert a single dataset split.
        
        Args:
            json_path: Path to the annotation JSON file
            split_name: Name of the split (train, val, test)
            
        Returns:
            ConversionStats with results
        """
        stats = ConversionStats()
        
        logger.info(f"Converting {json_path.name} ({split_name} split)")
        
        # Load annotations
        try:
            data = load_mmocr_json(json_path)
        except Exception as e:
            stats.errors.append(f"Failed to load {json_path}: {e}")
            return stats
        
        data_list = data.get("data_list", [])
        if not data_list:
            stats.errors.append(f"No data_list in {json_path}")
            return stats
        
        # Setup directories
        input_image_dir = self.detect_image_directory(json_path)
        output_image_dir = self.output_dir / "images"
        output_label_dir = self.output_dir / "labels"
        
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  Input images: {input_image_dir}")
        logger.info(f"  Output: {self.output_dir}")
        
        # Process images
        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for image_info in data_list:
                    future = executor.submit(
                        convert_single_image,
                        image_info,
                        input_image_dir,
                        output_image_dir,
                        output_label_dir,
                        self.copy_images,
                    )
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {split_name}"):
                    try:
                        success, num_poly, num_ignored = future.result()
                        if success:
                            stats.images_processed += 1
                            stats.polygons_converted += num_poly
                            stats.polygons_ignored += num_ignored
                        else:
                            stats.images_skipped += 1
                    except Exception as e:
                        stats.errors.append(str(e))
        else:
            for image_info in tqdm(data_list, desc=f"  {split_name}"):
                try:
                    success, num_poly, num_ignored = convert_single_image(
                        image_info,
                        input_image_dir,
                        output_image_dir,
                        output_label_dir,
                        self.copy_images,
                    )
                    if success:
                        stats.images_processed += 1
                        stats.polygons_converted += num_poly
                        stats.polygons_ignored += num_ignored
                    else:
                        stats.images_skipped += 1
                except Exception as e:
                    stats.errors.append(str(e))
        
        return stats
    
    def convert(self) -> ConversionStats:
        """
        Convert the entire dataset.
        
        Returns:
            ConversionStats with combined results
        """
        total_stats = ConversionStats()
        
        # Find annotation files
        json_files = self.find_annotation_files()
        
        if not json_files:
            total_stats.errors.append(f"No annotation files found in {self.input_dir}")
            return total_stats
        
        logger.info(f"Found {len(json_files)} annotation file(s)")
        
        # Convert each split
        for json_path in json_files:
            split_name = json_path.stem.replace("textdet_", "")
            stats = self.convert_split(json_path, split_name)
            
            # Merge stats
            total_stats.images_processed += stats.images_processed
            total_stats.images_skipped += stats.images_skipped
            total_stats.polygons_converted += stats.polygons_converted
            total_stats.polygons_ignored += stats.polygons_ignored
            total_stats.errors.extend(stats.errors)
        
        logger.info(f"Conversion complete:")
        logger.info(f"  Images processed: {total_stats.images_processed}")
        logger.info(f"  Images skipped: {total_stats.images_skipped}")
        logger.info(f"  Polygons converted: {total_stats.polygons_converted}")
        logger.info(f"  Polygons ignored: {total_stats.polygons_ignored}")
        
        if total_stats.errors:
            logger.warning(f"  Errors: {len(total_stats.errors)}")
        
        return total_stats


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    copy_images: bool = True,
    workers: int = 4,
) -> ConversionStats:
    """
    Convenience function to convert a dataset.
    
    Args:
        input_dir: MMOCR dataset directory
        output_dir: Output directory for YOLOv8 format
        copy_images: Whether to copy images
        workers: Number of parallel workers
        
    Returns:
        ConversionStats with results
    """
    converter = MMOCRToYOLOConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        copy_images=copy_images,
        workers=workers,
    )
    return converter.convert()


def main():
    """CLI entry point for format conversion."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Convert MMOCR format to YOLOv8-Seg format"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input MMOCR dataset directory"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for YOLOv8 format"
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Don't copy images, only create label files"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers"
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
    
    # Convert
    stats = convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copy_images=not args.no_copy_images,
        workers=args.workers,
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Conversion Summary:")
    print("=" * 50)
    print(f"  Images processed:   {stats.images_processed}")
    print(f"  Images skipped:     {stats.images_skipped}")
    print(f"  Polygons converted: {stats.polygons_converted}")
    print(f"  Polygons ignored:   {stats.polygons_ignored}")
    
    if stats.errors:
        print(f"\n  Errors ({len(stats.errors)}):")
        for error in stats.errors[:10]:
            print(f"    - {error}")
        if len(stats.errors) > 10:
            print(f"    ... and {len(stats.errors) - 10} more")
    
    return 0 if not stats.errors and stats.images_processed > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
