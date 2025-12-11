"""
Composition engine for combining text assets onto backgrounds.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils import (
    apply_color_transfer,
    crop_to_square,
    get_contrasting_color,
    load_image,
    resize_image,
    weighted_random_choice,
)
from .collision import CollisionDetector
from .extract_assets import AssetLibrary, WordAsset
from .ingest import DatasetInfo, Polygon, SamplePair
from .synth_generator import SynthGenerator
from .warp_engine import WarpEngine, WarpType

logger = logging.getLogger("engine.compose")


@dataclass
class PlacementInfo:
    """Information about a placed text instance."""
    asset_type: str
    source_dataset: str
    position: Tuple[int, int]
    scale: float
    warp_type: str
    polygon: Polygon


@dataclass
class CompositionResult:
    """Result of image composition."""
    image: np.ndarray
    polygons: List[Polygon]
    placements: List[PlacementInfo] = field(default_factory=list)


class Composer:
    """Composes text assets onto background images."""
    
    def __init__(
        self,
        output_size: Tuple[int, int],
        real_ratio: float = 0.5,
        scale_min: float = 0.03,
        scale_max: float = 0.25,
        edge_blur: float = 1.5,
        shadow_opacity: float = 0.4
    ):
        self.output_size = output_size
        self.real_ratio = real_ratio
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.edge_blur = edge_blur
        self.shadow_opacity = shadow_opacity
        
        self.warp_engine = WarpEngine(intensity="moderate")
    
    def compose(
        self,
        background: np.ndarray,
        original_polygons: List[Polygon],
        real_assets: AssetLibrary,
        synth_generator: SynthGenerator,
        fill_random: bool = True
    ) -> CompositionResult:
        """
        Compose text assets onto a background.
        
        Args:
            background: RGB background image
            original_polygons: Original polygon positions to fill first
            real_assets: Library of real word assets
            synth_generator: Synthetic text generator
            fill_random: Whether to fill random positions after originals
            
        Returns:
            CompositionResult with composed image and polygons
        """
        bg = self._prepare_background(background)
        h, w = bg.shape[:2]
        
        collision_detector = CollisionDetector((w, h))
        
        result_image = bg.copy()
        result_polygons = []
        placements = []
        
        all_real_assets = real_assets.get_all_assets()
        
        for polygon in original_polygons:
            use_real = random.random() < self.real_ratio and all_real_assets
            
            if use_real:
                asset = random.choice(all_real_assets)
                asset_image = asset.image.copy()
                asset_type = "real"
                source_dataset = asset.source_dataset
            else:
                x1, y1, x2, y2 = polygon.get_bbox((h, w))
                region = bg[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else bg[:10, :10]
                asset_image = synth_generator.generate_for_background(
                    region,
                    target_height=y2 - y1
                )
                asset_type = "synthetic"
                source_dataset = "generated"
            
            placed = self._place_at_polygon(
                result_image,
                asset_image,
                polygon,
                collision_detector
            )
            
            if placed:
                new_polygon, position, scale = placed
                result_polygons.append(new_polygon)
                
                placements.append(PlacementInfo(
                    asset_type=asset_type,
                    source_dataset=source_dataset,
                    position=position,
                    scale=scale,
                    warp_type="original_position",
                    polygon=new_polygon
                ))
        
        if fill_random:
            min_free_area = 0.1
            max_random_placements = 50
            
            for _ in range(max_random_placements):
                if collision_detector.get_free_area_ratio() < min_free_area:
                    break
                
                use_real = random.random() < self.real_ratio and all_real_assets
                
                if use_real:
                    asset = random.choice(all_real_assets)
                    asset_image = asset.image.copy()
                    asset_type = "real"
                    source_dataset = asset.source_dataset
                else:
                    asset_image = synth_generator.generate_single()
                    asset_type = "synthetic"
                    source_dataset = "generated"
                
                placed = self._place_random(
                    result_image,
                    asset_image,
                    collision_detector
                )
                
                if placed:
                    new_polygon, position, scale, warp_type = placed
                    result_polygons.append(new_polygon)
                    
                    placements.append(PlacementInfo(
                        asset_type=asset_type,
                        source_dataset=source_dataset,
                        position=position,
                        scale=scale,
                        warp_type=warp_type,
                        polygon=new_polygon
                    ))
        
        return CompositionResult(
            image=result_image,
            polygons=result_polygons,
            placements=placements
        )
    
    def _prepare_background(self, background: np.ndarray) -> np.ndarray:
        """Prepare background image (resize to output size)."""
        bg = crop_to_square(background)
        
        return resize_image(bg, self.output_size)
    
    def _scale_asset(
        self,
        asset: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, float]:
        """Scale asset to appropriate size."""
        h, w = asset.shape[:2]
        
        if target_size:
            target_h = target_size[1] - target_size[0]
            scale = target_h / h
        else:
            min_dim = int(self.output_size[0] * self.scale_min)
            max_dim = int(self.output_size[0] * self.scale_max)
            
            bias_min = int(self.output_size[0] * 0.08)
            bias_max = int(self.output_size[0] * 0.15)
            
            if random.random() < 0.6:
                target_h = random.randint(bias_min, bias_max)
            else:
                target_h = random.randint(min_dim, max_dim)
            
            scale = target_h / h
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w < 5 or new_h < 5:
            return asset, 1.0
        
        scaled = cv2.resize(
            asset,
            (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        return scaled, scale
    
    def _apply_warp(
        self,
        asset: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Apply random warp to asset."""
        warp_type = random.choice(list(WarpType))
        
        warped_image, warped_mask = self.warp_engine.warp(asset, warp_type)
        
        return warped_image, warped_mask, warp_type.value
    
    def _blend_asset(
        self,
        background: np.ndarray,
        asset: np.ndarray,
        position: Tuple[int, int]
    ) -> np.ndarray:
        """Blend asset onto background with color matching and shadow."""
        x, y = position
        asset_h, asset_w = asset.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + asset_w)
        y2 = min(bg_h, y + asset_h)
        
        if x2 <= x1 or y2 <= y1:
            return background
        
        src_x1 = x1 - x
        src_y1 = y1 - y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        asset_region = asset[src_y1:src_y2, src_x1:src_x2]
        bg_region = background[y1:y2, x1:x2]
        
        if asset_region.shape[2] == 4:
            alpha = asset_region[:, :, 3:4].astype(np.float32) / 255.0
            asset_rgb = asset_region[:, :, :3]
        else:
            alpha = np.ones((*asset_region.shape[:2], 1), dtype=np.float32)
            asset_rgb = asset_region
        
        if self.edge_blur > 0:
            alpha_blurred = cv2.GaussianBlur(
                alpha,
                (0, 0),
                self.edge_blur
            )
            if alpha_blurred.ndim == 2:
                alpha_blurred = alpha_blurred[:, :, np.newaxis]
            alpha = alpha_blurred
        
        if self.shadow_opacity > 0:
            shadow_offset = (3, 3)
            shadow = np.zeros_like(bg_region, dtype=np.float32)
            
            sx, sy = shadow_offset
            sh, sw = asset_region.shape[:2]
            
            if sy < sh and sx < sw:
                shadow_alpha = alpha[:-sy if sy > 0 else sh, :-sx if sx > 0 else sw]
                shadow_region = shadow[sy:, sx:]
                
                min_h = min(shadow_alpha.shape[0], shadow_region.shape[0])
                min_w = min(shadow_alpha.shape[1], shadow_region.shape[1])
                
                shadow[sy:sy+min_h, sx:sx+min_w] = (
                    shadow_alpha[:min_h, :min_w] * self.shadow_opacity * 50
                )
            
            shadow = cv2.GaussianBlur(shadow, (0, 0), 2)
            bg_region = np.clip(bg_region.astype(np.float32) - shadow, 0, 255)
        
        blended = (
            alpha * asset_rgb.astype(np.float32) +
            (1 - alpha) * bg_region.astype(np.float32)
        )
        
        result = background.copy()
        result[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return result
    
    def _place_at_polygon(
        self,
        background: np.ndarray,
        asset: np.ndarray,
        polygon: Polygon,
        collision_detector: CollisionDetector
    ) -> Optional[Tuple[Polygon, Tuple[int, int], float]]:
        """Place asset at a specific polygon location."""
        h, w = background.shape[:2]
        
        x1, y1, x2, y2 = polygon.get_bbox((h, w))
        target_h = y2 - y1
        target_w = x2 - x1
        
        if target_h < 10 or target_w < 10:
            return None
        
        scale = target_h / asset.shape[0]
        new_w = int(asset.shape[1] * scale)
        new_h = int(asset.shape[0] * scale)
        
        if new_w < 5 or new_h < 5:
            return None
        
        scaled_asset = cv2.resize(
            asset,
            (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        warped_asset, warped_mask, _ = self._apply_warp(scaled_asset)
        
        if warped_asset.shape[2] == 4:
            mask = warped_asset[:, :, 3]
        else:
            mask = np.ones(warped_asset.shape[:2], dtype=np.uint8) * 255
        
        position = (x1, y1)
        
        if collision_detector.check_mask_collision(mask, position):
            return None
        
        background[:] = self._blend_asset(background, warped_asset, position)
        
        collision_detector.add_mask(mask, position)
        
        new_polygon = self._mask_to_polygon(mask, position, (h, w))
        
        if new_polygon is None:
            return None
        
        return new_polygon, position, scale
    
    def _place_random(
        self,
        background: np.ndarray,
        asset: np.ndarray,
        collision_detector: CollisionDetector
    ) -> Optional[Tuple[Polygon, Tuple[int, int], float, str]]:
        """Place asset at a random valid position."""
        scaled_asset, scale = self._scale_asset(asset)
        
        warped_asset, warped_mask, warp_type = self._apply_warp(scaled_asset)
        
        if warped_asset.shape[2] == 4:
            mask = warped_asset[:, :, 3]
        else:
            mask = np.ones(warped_asset.shape[:2], dtype=np.uint8) * 255
        
        position = collision_detector.find_valid_position(mask)
        
        if position is None:
            position = collision_detector.find_valid_position_grid(mask)
        
        if position is None:
            return None
        
        h, w = background.shape[:2]
        background[:] = self._blend_asset(background, warped_asset, position)
        
        collision_detector.add_mask(mask, position)
        
        new_polygon = self._mask_to_polygon(mask, position, (h, w))
        
        if new_polygon is None:
            return None
        
        return new_polygon, position, scale, warp_type
    
    def _mask_to_polygon(
        self,
        mask: np.ndarray,
        position: Tuple[int, int],
        image_size: Tuple[int, int]
    ) -> Optional[Polygon]:
        """Convert a placed mask to a normalized polygon."""
        h, w = image_size
        x, y = position
        
        mask_binary = (mask > 127).astype(np.uint8)
        
        contours, _ = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 100:
            return None
        
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) < 3:
            return None
        
        points = [
            ((p[0][0] + x) / w, (p[0][1] + y) / h)
            for p in approx
        ]
        
        points = [(max(0, min(1, px)), max(0, min(1, py))) for px, py in points]
        
        return Polygon(class_id=0, points=points)
