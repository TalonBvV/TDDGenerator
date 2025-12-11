"""
LaMa inpainting module for cleaning backgrounds by removing text regions.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..models.lama_loader import LamaModel
from ..utils import batch_iterator, load_image, polygon_to_mask, save_image
from .ingest import DatasetInfo, Polygon, SamplePair

logger = logging.getLogger("engine.inpaint_lama")


class LamaInpainter:
    """Handles background cleaning using LaMa inpainting."""
    
    def __init__(
        self,
        lama_model: LamaModel,
        output_dir: Path,
        batch_size: int = 8
    ):
        self.lama_model = lama_model
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        
        self.backgrounds_dir = self.output_dir / "assets" / "backgrounds"
        self.backgrounds_dir.mkdir(parents=True, exist_ok=True)
    
    def create_inpaint_mask(
        self,
        polygons: List[Polygon],
        image_size: Tuple[int, int],
        dilation: int = 5
    ) -> np.ndarray:
        """
        Create inpainting mask from polygons.
        
        Args:
            polygons: List of polygon annotations
            image_size: (height, width) of the image
            dilation: Pixels to dilate the mask for better inpainting
            
        Returns:
            Binary mask where white (255) indicates regions to inpaint
        """
        h, w = image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for polygon in polygons:
            poly_mask = polygon_to_mask(polygon.points, image_size)
            mask = np.maximum(mask, poly_mask)
        
        if dilation > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel)
        
        return mask
    
    def clean_background(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> np.ndarray:
        """
        Remove text regions from an image using LaMa inpainting.
        
        Args:
            image: RGB numpy array
            polygons: List of polygon annotations marking text regions
            
        Returns:
            Inpainted RGB numpy array with text removed
        """
        if not polygons:
            return image
        
        h, w = image.shape[:2]
        mask = self.create_inpaint_mask(polygons, (h, w))
        
        if mask.max() == 0:
            return image
        
        return self.lama_model.inpaint_single(image, mask)
    
    def process_sample(
        self,
        sample: SamplePair,
        dataset_name: str,
        save_background: bool = True
    ) -> Optional[np.ndarray]:
        """
        Process a single sample to create a clean background.
        
        Args:
            sample: Image-label pair
            dataset_name: Name of the source dataset
            save_background: Whether to save the background to disk
            
        Returns:
            Clean background image or None if processing fails
        """
        try:
            image = load_image(sample.image_path)
        except Exception as e:
            logger.error(f"Failed to load image {sample.image_path}: {e}")
            return None
        
        if not sample.polygons:
            sample.load_polygons()
        
        try:
            clean_bg = self.clean_background(image, sample.polygons)
        except Exception as e:
            logger.error(f"Failed to inpaint {sample.image_path}: {e}")
            return None
        
        if save_background:
            dataset_dir = self.backgrounds_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            bg_path = dataset_dir / f"bg_{sample.image_path.stem}.png"
            save_image(clean_bg, bg_path)
        
        return clean_bg
    
    def process_dataset(
        self,
        dataset: DatasetInfo,
        save_backgrounds: bool = True
    ) -> List[Path]:
        """
        Process all samples in a dataset to create clean backgrounds.
        
        Args:
            dataset: Dataset information
            save_backgrounds: Whether to save backgrounds to disk
            
        Returns:
            List of paths to saved background images
        """
        logger.info(f"Processing backgrounds for dataset: {dataset.name}")
        
        saved_paths = []
        
        for batch in batch_iterator(dataset.samples, self.batch_size):
            images = []
            masks = []
            valid_samples = []
            
            for sample in batch:
                try:
                    image = load_image(sample.image_path)
                    
                    if not sample.polygons:
                        sample.load_polygons()
                    
                    h, w = image.shape[:2]
                    mask = self.create_inpaint_mask(sample.polygons, (h, w))
                    
                    images.append(image)
                    masks.append(mask)
                    valid_samples.append(sample)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare {sample.image_path}: {e}")
            
            if not images:
                continue
            
            try:
                clean_bgs = self.lama_model.inpaint_batch(images, masks)
            except Exception as e:
                logger.error(f"Batch inpainting failed: {e}")
                clean_bgs = []
                for img, msk in zip(images, masks):
                    try:
                        clean_bgs.append(self.lama_model.inpaint_single(img, msk))
                    except Exception:
                        clean_bgs.append(img)
            
            if save_backgrounds:
                dataset_dir = self.backgrounds_dir / dataset.name
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                for sample, clean_bg in zip(valid_samples, clean_bgs):
                    bg_path = dataset_dir / f"bg_{sample.image_path.stem}.png"
                    save_image(clean_bg, bg_path)
                    saved_paths.append(bg_path)
        
        logger.info(f"Processed {len(saved_paths)} backgrounds for '{dataset.name}'")
        
        return saved_paths
    
    def process_all_datasets(
        self,
        datasets: List[DatasetInfo],
        save_backgrounds: bool = True
    ) -> dict:
        """
        Process all datasets and create clean backgrounds.
        
        Args:
            datasets: List of dataset information
            save_backgrounds: Whether to save backgrounds to disk
            
        Returns:
            Dictionary mapping dataset names to lists of background paths
        """
        all_backgrounds = {}
        
        for dataset in datasets:
            paths = self.process_dataset(dataset, save_backgrounds)
            all_backgrounds[dataset.name] = paths
        
        total = sum(len(paths) for paths in all_backgrounds.values())
        logger.info(f"Total backgrounds processed: {total}")
        
        return all_backgrounds
    
    def load_existing_backgrounds(
        self,
        datasets: List[DatasetInfo]
    ) -> dict:
        """
        Load previously processed backgrounds from disk.
        
        Args:
            datasets: List of dataset information
            
        Returns:
            Dictionary mapping dataset names to lists of background paths
        """
        all_backgrounds = {}
        
        for dataset in datasets:
            dataset_dir = self.backgrounds_dir / dataset.name
            
            if not dataset_dir.exists():
                all_backgrounds[dataset.name] = []
                continue
            
            paths = sorted(dataset_dir.glob("bg_*.png"))
            all_backgrounds[dataset.name] = paths
        
        total = sum(len(paths) for paths in all_backgrounds.values())
        logger.info(f"Loaded {total} existing backgrounds")
        
        return all_backgrounds
    
    def inpaint_region(
        self,
        image: np.ndarray,
        polygon: Polygon
    ) -> np.ndarray:
        """
        Inpaint a single polygon region in an image.
        
        Args:
            image: RGB numpy array
            polygon: Single polygon to inpaint
            
        Returns:
            Image with the polygon region inpainted
        """
        h, w = image.shape[:2]
        mask = polygon_to_mask(polygon.points, (h, w))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, kernel)
        
        return self.lama_model.inpaint_single(image, mask)
