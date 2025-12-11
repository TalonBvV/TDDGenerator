"""
Asset extraction module for extracting word assets from annotated images.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.rmbg_loader import RMBGModel
from ..utils import (
    batch_iterator,
    crop_with_mask,
    get_bbox_from_polygon,
    load_image,
    polygon_to_mask,
    save_image_pil,
)
from .ingest import DatasetInfo, Polygon, SamplePair

logger = logging.getLogger("engine.extract_assets")


@dataclass
class WordAsset:
    """Represents an extracted word asset."""
    image: np.ndarray
    source_dataset: str
    source_image: str
    original_polygon: Polygon
    bbox: Tuple[int, int, int, int]
    asset_path: Optional[Path] = None
    
    @property
    def width(self) -> int:
        return self.image.shape[1]
    
    @property
    def height(self) -> int:
        return self.image.shape[0]
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


@dataclass
class AssetLibrary:
    """Collection of word assets organized by source dataset."""
    assets: Dict[str, List[WordAsset]] = field(default_factory=dict)
    
    def add_asset(self, asset: WordAsset) -> None:
        """Add an asset to the library."""
        if asset.source_dataset not in self.assets:
            self.assets[asset.source_dataset] = []
        self.assets[asset.source_dataset].append(asset)
    
    def get_all_assets(self) -> List[WordAsset]:
        """Get all assets from all datasets."""
        all_assets = []
        for dataset_assets in self.assets.values():
            all_assets.extend(dataset_assets)
        return all_assets
    
    def get_dataset_assets(self, dataset_name: str) -> List[WordAsset]:
        """Get assets from a specific dataset."""
        return self.assets.get(dataset_name, [])
    
    @property
    def total_count(self) -> int:
        return sum(len(assets) for assets in self.assets.values())
    
    def get_dataset_weights(self) -> Dict[str, float]:
        """Get inverse weights for balanced sampling."""
        counts = {name: len(assets) for name, assets in self.assets.items()}
        total = sum(counts.values())
        
        if total == 0:
            return {name: 1.0 for name in self.assets}
        
        weights = {name: total / max(count, 1) for name, count in counts.items()}
        total_weight = sum(weights.values())
        
        return {name: w / total_weight for name, w in weights.items()}


class AssetExtractor:
    """Extracts word assets from annotated images using RMBG for alpha matting."""
    
    def __init__(
        self,
        rmbg_model: RMBGModel,
        output_dir: Path,
        batch_size: int = 8,
        min_asset_size: int = 10
    ):
        self.rmbg_model = rmbg_model
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.min_asset_size = min_asset_size
        
        self.real_text_dir = self.output_dir / "assets" / "real_text"
        self.real_text_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_words(self, sample: SamplePair, dataset_name: str) -> List[WordAsset]:
        """Extract word assets from a single sample."""
        try:
            image = load_image(sample.image_path)
        except Exception as e:
            logger.error(f"Failed to load image {sample.image_path}: {e}")
            return []
        
        h, w = image.shape[:2]
        assets = []
        
        crops = []
        crop_info = []
        
        for polygon in sample.polygons:
            try:
                mask = polygon_to_mask(polygon.points, (h, w))
                bbox = polygon.get_bbox((h, w))
                x1, y1, x2, y2 = bbox
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                if crop_w < self.min_asset_size or crop_h < self.min_asset_size:
                    continue
                
                cropped = image[y1:y2, x1:x2].copy()
                
                crops.append(cropped)
                crop_info.append({
                    "polygon": polygon,
                    "bbox": bbox,
                    "source_image": sample.image_path.stem
                })
                
            except Exception as e:
                logger.warning(f"Failed to extract polygon from {sample.image_path}: {e}")
                continue
        
        if not crops:
            return assets
        
        try:
            rgba_crops = self.rmbg_model.process_batch(crops)
        except Exception as e:
            logger.error(f"RMBG batch processing failed: {e}")
            rgba_crops = []
            for crop in crops:
                try:
                    rgba = self.rmbg_model.process_single(crop)
                    rgba_crops.append(rgba)
                except Exception:
                    rgba_crops.append(None)
        
        for i, rgba in enumerate(rgba_crops):
            if rgba is None:
                continue
            
            info = crop_info[i]
            
            asset = WordAsset(
                image=rgba,
                source_dataset=dataset_name,
                source_image=info["source_image"],
                original_polygon=info["polygon"],
                bbox=info["bbox"]
            )
            
            assets.append(asset)
        
        return assets
    
    def process_dataset(
        self,
        dataset: DatasetInfo,
        save_assets: bool = True
    ) -> AssetLibrary:
        """Process an entire dataset and extract all word assets."""
        logger.info(f"Extracting assets from dataset: {dataset.name}")
        
        library = AssetLibrary()
        
        dataset_output_dir = self.real_text_dir / dataset.name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        asset_counter = 0
        
        for batch in batch_iterator(dataset.samples, self.batch_size):
            for sample in batch:
                assets = self.extract_words(sample, dataset.name)
                
                for asset in assets:
                    if save_assets:
                        asset_path = dataset_output_dir / f"word_{asset_counter:06d}.png"
                        save_image_pil(asset.image, asset_path)
                        asset.asset_path = asset_path
                        asset_counter += 1
                    
                    library.add_asset(asset)
        
        logger.info(
            f"Extracted {library.total_count} assets from dataset '{dataset.name}'"
        )
        
        return library
    
    def process_all_datasets(
        self,
        datasets: List[DatasetInfo],
        save_assets: bool = True
    ) -> AssetLibrary:
        """Process all datasets and merge into a single asset library."""
        combined_library = AssetLibrary()
        
        for dataset in datasets:
            library = self.process_dataset(dataset, save_assets)
            
            for dataset_name, assets in library.assets.items():
                for asset in assets:
                    combined_library.add_asset(asset)
        
        logger.info(f"Total extracted assets: {combined_library.total_count}")
        
        return combined_library
    
    def load_existing_assets(self, datasets: List[DatasetInfo]) -> AssetLibrary:
        """Load previously extracted assets from disk."""
        library = AssetLibrary()
        
        for dataset in datasets:
            dataset_dir = self.real_text_dir / dataset.name
            
            if not dataset_dir.exists():
                continue
            
            for asset_path in sorted(dataset_dir.glob("word_*.png")):
                try:
                    image = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        continue
                    
                    if image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                    
                    asset = WordAsset(
                        image=image,
                        source_dataset=dataset.name,
                        source_image="",
                        original_polygon=Polygon(class_id=0, points=[]),
                        bbox=(0, 0, image.shape[1], image.shape[0]),
                        asset_path=asset_path
                    )
                    
                    library.add_asset(asset)
                    
                except Exception as e:
                    logger.warning(f"Failed to load asset {asset_path}: {e}")
        
        logger.info(f"Loaded {library.total_count} existing assets")
        
        return library
