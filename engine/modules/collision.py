"""
Collision detection module for pixel-level overlap checking.
"""

import logging
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .ingest import Polygon

logger = logging.getLogger("engine.collision")


class CollisionDetector:
    """Pixel-level collision detection for text placement."""
    
    def __init__(self, canvas_size: Tuple[int, int]):
        """
        Initialize collision detector.
        
        Args:
            canvas_size: (width, height) of the canvas
        """
        self.width, self.height = canvas_size
        self.occupancy_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.polygons: List[Polygon] = []
    
    def reset(self) -> None:
        """Reset the occupancy mask and polygon list."""
        self.occupancy_mask.fill(0)
        self.polygons.clear()
    
    def add_polygon(self, polygon: Polygon) -> None:
        """
        Add a polygon to the occupancy mask.
        
        Args:
            polygon: Polygon with normalized coordinates
        """
        mask = self._polygon_to_mask(polygon)
        self.occupancy_mask = np.maximum(self.occupancy_mask, mask)
        self.polygons.append(polygon)
    
    def add_mask(self, mask: np.ndarray, position: Tuple[int, int]) -> None:
        """
        Add a binary mask to the occupancy mask at a position.
        
        Args:
            mask: Binary mask (0 or 255)
            position: (x, y) top-left position
        """
        x, y = position
        h, w = mask.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + w)
        y2 = min(self.height, y + h)
        
        src_x1 = x1 - x
        src_y1 = y1 - y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1:
            mask_region = mask[src_y1:src_y2, src_x1:src_x2]
            self.occupancy_mask[y1:y2, x1:x2] = np.maximum(
                self.occupancy_mask[y1:y2, x1:x2],
                mask_region
            )
    
    def check_collision(self, polygon: Polygon) -> bool:
        """
        Check if a polygon collides with existing occupied regions.
        
        Args:
            polygon: Polygon with normalized coordinates
            
        Returns:
            True if collision detected, False otherwise
        """
        mask = self._polygon_to_mask(polygon)
        
        overlap = np.logical_and(self.occupancy_mask > 0, mask > 0)
        
        return np.any(overlap)
    
    def check_mask_collision(
        self,
        mask: np.ndarray,
        position: Tuple[int, int]
    ) -> bool:
        """
        Check if a mask at a position collides with occupied regions.
        
        Args:
            mask: Binary mask
            position: (x, y) top-left position
            
        Returns:
            True if collision detected, False otherwise
        """
        x, y = position
        h, w = mask.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + w)
        y2 = min(self.height, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return True
        
        src_x1 = x1 - x
        src_y1 = y1 - y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        mask_region = mask[src_y1:src_y2, src_x1:src_x2]
        occupancy_region = self.occupancy_mask[y1:y2, x1:x2]
        
        overlap = np.logical_and(occupancy_region > 0, mask_region > 0)
        
        return np.any(overlap)
    
    def check_bounds(
        self,
        mask: np.ndarray,
        position: Tuple[int, int]
    ) -> bool:
        """
        Check if a mask at a position is within canvas bounds.
        
        Args:
            mask: Binary mask
            position: (x, y) top-left position
            
        Returns:
            True if fully within bounds, False otherwise
        """
        x, y = position
        h, w = mask.shape[:2]
        
        return (
            x >= 0 and
            y >= 0 and
            x + w <= self.width and
            y + h <= self.height
        )
    
    def find_valid_position(
        self,
        mask: np.ndarray,
        max_attempts: int = 100,
        margin: int = 10
    ) -> Optional[Tuple[int, int]]:
        """
        Find a valid position using downsampled spatial indexing for speed.
        """
        h, w = mask.shape[:2]
        max_x = self.width - w - margin
        max_y = self.height - h - margin
        
        if max_x < margin or max_y < margin:
            return None
        
        free_ratio = self.get_free_area_ratio()
        if free_ratio < 0.05:
            return None
        
        if free_ratio > 0.5:
            for _ in range(max_attempts):
                x = random.randint(margin, max_x)
                y = random.randint(margin, max_y)
                if not self.check_mask_collision(mask, (x, y)):
                    return (x, y)
            return None
        
        scale = 4
        small_occ = cv2.resize(self.occupancy_mask, 
                               (self.width // scale, self.height // scale),
                               interpolation=cv2.INTER_NEAREST)
        small_mask = cv2.resize(mask, (w // scale + 1, h // scale + 1),
                                interpolation=cv2.INTER_NEAREST)
        
        free_y, free_x = np.where(small_occ == 0)
        if len(free_x) == 0:
            return None
        
        candidates = list(zip(free_x * scale, free_y * scale))
        random.shuffle(candidates)
        
        for cx, cy in candidates[:max_attempts * 2]:
            x = max(margin, min(max_x, cx - w // 2))
            y = max(margin, min(max_y, cy - h // 2))
            if not self.check_mask_collision(mask, (x, y)):
                return (x, y)
        
        return None
    
    def find_valid_position_grid(
        self,
        mask: np.ndarray,
        grid_step: int = 20,
        margin: int = 10
    ) -> Optional[Tuple[int, int]]:
        """
        Find a valid position using grid search (more thorough but slower).
        
        Args:
            mask: Binary mask to place
            grid_step: Step size for grid search
            margin: Minimum margin from canvas edges
            
        Returns:
            (x, y) position or None if no valid position found
        """
        h, w = mask.shape[:2]
        
        positions = []
        for y in range(margin, self.height - h - margin, grid_step):
            for x in range(margin, self.width - w - margin, grid_step):
                if not self.check_mask_collision(mask, (x, y)):
                    positions.append((x, y))
        
        if positions:
            return random.choice(positions)
        
        return None
    
    def get_free_area_ratio(self) -> float:
        """
        Get the ratio of free (unoccupied) area on the canvas.
        
        Returns:
            Ratio of free pixels (0.0 to 1.0)
        """
        total_pixels = self.width * self.height
        occupied_pixels = np.sum(self.occupancy_mask > 0)
        
        return 1.0 - (occupied_pixels / total_pixels)
    
    def _polygon_to_mask(self, polygon: Polygon) -> np.ndarray:
        """Convert a polygon to a binary mask."""
        points = np.array([
            [int(x * self.width), int(y * self.height)]
            for x, y in polygon.points
        ], dtype=np.int32)
        
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def get_polygon_position(
        self,
        polygon: Polygon
    ) -> Tuple[int, int, int, int]:
        """
        Get bounding box of a polygon in pixel coordinates.
        
        Args:
            polygon: Polygon with normalized coordinates
            
        Returns:
            (x1, y1, x2, y2) bounding box
        """
        xs = [int(x * self.width) for x, y in polygon.points]
        ys = [int(y * self.height) for x, y in polygon.points]
        
        return min(xs), min(ys), max(xs), max(ys)
    
    def visualize(self) -> np.ndarray:
        """
        Create a visualization of the current occupancy mask.
        
        Returns:
            RGB image showing occupied regions
        """
        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        vis[:, :, 0] = self.occupancy_mask
        vis[:, :, 1] = self.occupancy_mask // 2
        
        return vis
