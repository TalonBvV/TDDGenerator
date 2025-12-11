"""
Warp engine for applying geometric deformations to text assets.
Optimized with vectorized numpy operations.
"""

import logging
import random
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .ingest import Polygon

logger = logging.getLogger("engine.warp_engine")


class WarpType(Enum):
    """Available warp types."""
    NONE = "none"
    PERSPECTIVE = "perspective"
    CURVE = "curve"
    ARC = "arc"
    SINE_WAVE = "sine_wave"
    CIRCULAR = "circular"
    SPIRAL = "spiral"
    FREEFORM_POLYGON = "freeform_polygon"


class WarpEngine:
    """Applies geometric deformations to images and masks."""
    
    WARP_TYPES = [
        WarpType.PERSPECTIVE,
        WarpType.CURVE,
        WarpType.ARC,
        WarpType.SINE_WAVE,
        WarpType.CIRCULAR,
        WarpType.SPIRAL,
        WarpType.FREEFORM_POLYGON,
    ]
    
    def __init__(self, intensity: str = "moderate"):
        self.intensity = intensity
        self._set_intensity_params()
    
    def _set_intensity_params(self) -> None:
        """Set parameters based on intensity level."""
        if self.intensity == "mild":
            self.perspective_range = 0.05
            self.curve_range = 0.1
            self.wave_amplitude = 0.05
            self.wave_frequency = 1
            self.arc_angle_range = (5, 20)
        elif self.intensity == "extreme":
            self.perspective_range = 0.2
            self.curve_range = 0.4
            self.wave_amplitude = 0.2
            self.wave_frequency = 4
            self.arc_angle_range = (20, 60)
        else:
            self.perspective_range = 0.1
            self.curve_range = 0.2
            self.wave_amplitude = 0.1
            self.wave_frequency = 2
            self.arc_angle_range = (10, 40)
    
    def warp(
        self,
        image: np.ndarray,
        warp_type: Optional[WarpType] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply warp to image.
        
        Args:
            image: RGBA numpy array
            warp_type: Type of warp to apply (random if None)
            
        Returns:
            Tuple of (warped image, warped mask)
        """
        if warp_type is None or warp_type == WarpType.NONE:
            warp_type = random.choice(self.WARP_TYPES)
        
        if image.shape[2] == 4:
            mask = image[:, :, 3]
        else:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        if warp_type == WarpType.PERSPECTIVE:
            return self._perspective_warp(image, mask)
        elif warp_type == WarpType.CURVE:
            return self._curve_warp(image, mask)
        elif warp_type == WarpType.ARC:
            return self._arc_warp(image, mask)
        elif warp_type == WarpType.SINE_WAVE:
            return self._sine_wave_warp(image, mask)
        elif warp_type == WarpType.CIRCULAR:
            return self._circular_warp(image, mask)
        elif warp_type == WarpType.SPIRAL:
            return self._spiral_warp(image, mask)
        elif warp_type == WarpType.FREEFORM_POLYGON:
            return self._freeform_warp(image, mask)
        else:
            return image, mask
    
    def _perspective_warp(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply perspective transformation."""
        h, w = image.shape[:2]
        
        max_offset = int(min(h, w) * self.perspective_range)
        
        src_pts = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])
        
        dst_pts = np.float32([
            [random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(0, max_offset), h - random.randint(0, max_offset)],
            [random.randint(0, max_offset), h - random.randint(0, max_offset)]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        warped_image = cv2.warpPerspective(
            image, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        warped_mask = cv2.warpPerspective(
            mask, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped_image, warped_mask
    
    def _curve_warp(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply bezier curve warp - vectorized."""
        h, w = image.shape[:2]
        
        curve_strength = self.curve_range * h
        direction = random.choice([-1, 1])
        
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        t = xx / w
        y_normalized = yy / h
        curve_offset = curve_strength * np.sin(np.pi * t) * (1 - 2 * np.abs(y_normalized - 0.5))
        
        x_map = xx.astype(np.float32)
        y_map = (yy + curve_offset * direction).astype(np.float32)
        
        warped_image = cv2.remap(
            image, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        warped_mask = cv2.remap(
            mask, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped_image, warped_mask
    
    def _arc_warp(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply arc/bend warp using proper cylindrical projection.
        Improved calculation for smoother curves.
        """
        h, w = image.shape[:2]
        
        arc_angle_deg = random.uniform(*self.arc_angle_range)
        arc_angle = arc_angle_deg * (np.pi / 180)
        direction = random.choice([-1, 1])
        
        if arc_angle < 0.01:
            return image, mask
        
        radius = (w / 2) / np.sin(arc_angle / 2)
        
        center_y = h / 2 + direction * radius
        
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        normalized_x = (xx - w / 2) / (w / 2)
        theta = normalized_x * (arc_angle / 2)
        
        new_x = w / 2 + radius * np.sin(theta)
        
        arc_y_offset = radius * (1 - np.cos(theta)) * (-direction)
        
        y_from_center = yy - h / 2
        scale_factor = np.cos(theta)
        scale_factor = np.clip(scale_factor, 0.5, 1.0)
        
        new_y = h / 2 + y_from_center * scale_factor + arc_y_offset
        
        x_map = new_x.astype(np.float32)
        y_map = new_y.astype(np.float32)
        
        warped_image = cv2.remap(
            image, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        warped_mask = cv2.remap(
            mask, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped_image, warped_mask
    
    def _sine_wave_warp(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sine wave distortion - vectorized."""
        h, w = image.shape[:2]
        
        amplitude = self.wave_amplitude * h
        frequency = self.wave_frequency
        phase = random.uniform(0, 2 * np.pi)
        
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        offset = amplitude * np.sin(2 * np.pi * frequency * xx / w + phase)
        
        x_map = xx.astype(np.float32)
        y_map = (yy + offset).astype(np.float32)
        
        warped_image = cv2.remap(
            image, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        warped_mask = cv2.remap(
            mask, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped_image, warped_mask
    
    def _circular_warp(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply circular/cylindrical warp - vectorized."""
        h, w = image.shape[:2]
        
        radius = random.uniform(max(w, 200), max(w * 2, 800))
        
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        center_x = w / 2
        dx = xx - center_x
        
        angle = np.clip(dx / radius, -np.pi/3, np.pi/3)
        
        new_x = center_x + radius * np.sin(angle)
        scale = np.cos(angle)
        new_y = h / 2 + (yy - h / 2) * scale
        
        x_map = new_x.astype(np.float32)
        y_map = new_y.astype(np.float32)
        
        warped_image = cv2.remap(
            image, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        warped_mask = cv2.remap(
            mask, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped_image, warped_mask
    
    def _spiral_warp(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spiral/twist warp - vectorized."""
        h, w = image.shape[:2]
        
        turns = random.uniform(0.1, 0.5)
        tightness = random.uniform(0.8, 1.2)
        
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        center_x, center_y = w / 2, h / 2
        max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
        
        dx = xx - center_x
        dy = yy - center_y
        radius = np.sqrt(dx ** 2 + dy ** 2)
        
        angle = np.arctan2(dy, dx)
        twist = turns * 2 * np.pi * np.power(radius / max_radius, tightness)
        
        new_angle = angle + twist
        new_x = center_x + radius * np.cos(new_angle)
        new_y = center_y + radius * np.sin(new_angle)
        
        x_map = new_x.astype(np.float32)
        y_map = new_y.astype(np.float32)
        
        warped_image = cv2.remap(
            image, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        warped_mask = cv2.remap(
            mask, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped_image, warped_mask
    
    def _freeform_warp(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply freeform polygon deformation using spline interpolation."""
        h, w = image.shape[:2]
        
        grid_size = 4
        displacement = self.perspective_range * min(h, w)
        
        ctrl_x = np.zeros((grid_size + 1, grid_size + 1), dtype=np.float32)
        ctrl_y = np.zeros((grid_size + 1, grid_size + 1), dtype=np.float32)
        
        for gy in range(grid_size + 1):
            for gx in range(grid_size + 1):
                base_x = gx * w / grid_size
                base_y = gy * h / grid_size
                
                if 0 < gx < grid_size and 0 < gy < grid_size:
                    dx = random.uniform(-displacement, displacement)
                    dy = random.uniform(-displacement, displacement)
                else:
                    dx, dy = 0, 0
                
                ctrl_x[gy, gx] = base_x + dx
                ctrl_y[gy, gx] = base_y + dy
        
        grid_y = np.linspace(0, h, grid_size + 1)
        grid_x = np.linspace(0, w, grid_size + 1)
        
        spline_x = RectBivariateSpline(grid_y, grid_x, ctrl_x, kx=2, ky=2)
        spline_y = RectBivariateSpline(grid_y, grid_x, ctrl_y, kx=2, ky=2)
        
        y_coords = np.arange(h, dtype=np.float32)
        x_coords = np.arange(w, dtype=np.float32)
        
        x_map = spline_x(y_coords, x_coords).astype(np.float32)
        y_map = spline_y(y_coords, x_coords).astype(np.float32)
        
        warped_image = cv2.remap(
            image, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        warped_mask = cv2.remap(
            mask, x_map, y_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped_image, warped_mask
    
    def get_polygon_from_mask(
        self,
        mask: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Optional[Polygon]:
        """Extract polygon from warped mask."""
        h, w = image_size
        
        mask_binary = (mask > 127).astype(np.uint8)
        
        contours, _ = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) < 3:
            return None
        
        points = [(p[0][0] / w, p[0][1] / h) for p in approx]
        
        return Polygon(class_id=0, points=points)
