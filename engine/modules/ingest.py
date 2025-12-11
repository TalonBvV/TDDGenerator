"""
Dataset ingestion module for scanning and validating YOLOv8-Seg datasets.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("engine.ingest")


@dataclass
class Polygon:
    """Represents a polygon annotation with normalized coordinates."""
    class_id: int
    points: List[Tuple[float, float]]
    
    @classmethod
    def from_yolo_line(cls, line: str) -> "Polygon":
        """Parse a YOLOv8-Seg format line."""
        parts = line.strip().split()
        if len(parts) < 7:
            raise ValueError(f"Invalid YOLO line: {line}")
        
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        if len(coords) % 2 != 0:
            raise ValueError(f"Odd number of coordinates in line: {line}")
        
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        
        return cls(class_id=class_id, points=points)
    
    def to_yolo_line(self) -> str:
        """Convert to YOLOv8-Seg format line."""
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_id} {coords}"
    
    def get_bbox(self, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get bounding box in pixel coordinates."""
        h, w = image_size
        xs = [int(x * w) for x, y in self.points]
        ys = [int(y * h) for x, y in self.points]
        return min(xs), min(ys), max(xs), max(ys)
    
    def scale(self, scale_x: float, scale_y: float) -> "Polygon":
        """Scale polygon coordinates."""
        new_points = [(x * scale_x, y * scale_y) for x, y in self.points]
        return Polygon(class_id=self.class_id, points=new_points)
    
    def translate(self, dx: float, dy: float) -> "Polygon":
        """Translate polygon coordinates."""
        new_points = [(x + dx, y + dy) for x, y in self.points]
        return Polygon(class_id=self.class_id, points=new_points)


@dataclass
class SamplePair:
    """Represents an image-label pair."""
    image_path: Path
    label_path: Path
    polygons: List[Polygon] = field(default_factory=list)
    
    def load_polygons(self) -> None:
        """Load polygons from label file."""
        self.polygons = []
        
        if not self.label_path.exists():
            logger.warning(f"Label file not found: {self.label_path}")
            return
        
        with open(self.label_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    polygon = Polygon.from_yolo_line(line)
                    self.polygons.append(polygon)
                except ValueError as e:
                    logger.warning(
                        f"Error parsing line {line_num} in {self.label_path}: {e}"
                    )


@dataclass
class DatasetInfo:
    """Information about a single dataset."""
    name: str
    root_dir: Path
    image_dir: Path
    label_dir: Path
    samples: List[SamplePair] = field(default_factory=list)
    
    @property
    def num_samples(self) -> int:
        return len(self.samples)
    
    @property
    def num_polygons(self) -> int:
        return sum(len(s.polygons) for s in self.samples)


@dataclass
class ValidationReport:
    """Report from dataset validation."""
    valid: bool
    datasets: List[DatasetInfo]
    total_images: int
    total_polygons: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DatasetIngester:
    """Scans and validates YOLOv8-Seg datasets."""
    
    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.datasets: List[DatasetInfo] = []
    
    def scan(self) -> List[DatasetInfo]:
        """Scan the dataset directory for valid datasets."""
        logger.info(f"Scanning dataset directory: {self.dataset_dir}")
        
        self.datasets = []
        
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {self.dataset_dir}")
        
        for subdir in sorted(self.dataset_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            image_dir = subdir / "images"
            label_dir = subdir / "labels"
            
            if not image_dir.exists() or not label_dir.exists():
                logger.warning(
                    f"Skipping {subdir.name}: missing images/ or labels/ directory"
                )
                continue
            
            dataset = DatasetInfo(
                name=subdir.name,
                root_dir=subdir,
                image_dir=image_dir,
                label_dir=label_dir
            )
            
            self._scan_dataset(dataset)
            
            if dataset.num_samples > 0:
                self.datasets.append(dataset)
                logger.info(
                    f"Found dataset '{dataset.name}': "
                    f"{dataset.num_samples} samples, {dataset.num_polygons} polygons"
                )
            else:
                logger.warning(f"Skipping empty dataset: {dataset.name}")
        
        logger.info(f"Found {len(self.datasets)} valid datasets")
        return self.datasets
    
    def _scan_dataset(self, dataset: DatasetInfo) -> None:
        """Scan a single dataset for image-label pairs."""
        image_files = {}
        for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            for img_path in dataset.image_dir.glob(f"*{ext}"):
                image_files[img_path.stem] = img_path
            for img_path in dataset.image_dir.glob(f"*{ext.upper()}"):
                image_files[img_path.stem] = img_path
        
        for stem, img_path in sorted(image_files.items()):
            label_path = dataset.label_dir / f"{stem}.txt"
            
            if not label_path.exists():
                logger.debug(f"No label file for {img_path.name}")
                continue
            
            sample = SamplePair(image_path=img_path, label_path=label_path)
            sample.load_polygons()
            
            if sample.polygons:
                dataset.samples.append(sample)
    
    def validate(self) -> ValidationReport:
        """Validate all scanned datasets."""
        if not self.datasets:
            self.scan()
        
        errors = []
        warnings = []
        total_images = 0
        total_polygons = 0
        
        for dataset in self.datasets:
            total_images += dataset.num_samples
            total_polygons += dataset.num_polygons
            
            for sample in dataset.samples:
                if not sample.image_path.exists():
                    errors.append(f"Missing image: {sample.image_path}")
                    continue
                
                for i, polygon in enumerate(sample.polygons):
                    for x, y in polygon.points:
                        if not (0 <= x <= 1 and 0 <= y <= 1):
                            warnings.append(
                                f"Out-of-bounds coordinates in {sample.label_path} "
                                f"polygon {i}: ({x}, {y})"
                            )
                    
                    if len(polygon.points) < 3:
                        warnings.append(
                            f"Degenerate polygon in {sample.label_path} "
                            f"polygon {i}: only {len(polygon.points)} points"
                        )
        
        valid = len(errors) == 0 and total_images > 0
        
        report = ValidationReport(
            valid=valid,
            datasets=self.datasets,
            total_images=total_images,
            total_polygons=total_polygons,
            errors=errors,
            warnings=warnings
        )
        
        if errors:
            logger.error(f"Validation found {len(errors)} errors")
        if warnings:
            logger.warning(f"Validation found {len(warnings)} warnings")
        
        logger.info(
            f"Validation complete: {total_images} images, {total_polygons} polygons"
        )
        
        return report
    
    def get_dataset_weights(self) -> List[float]:
        """Get inverse weights for balanced sampling (prioritize underrepresented)."""
        if not self.datasets:
            return []
        
        counts = [ds.num_samples for ds in self.datasets]
        total = sum(counts)
        
        if total == 0:
            return [1.0] * len(self.datasets)
        
        weights = [total / max(c, 1) for c in counts]
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return weights
