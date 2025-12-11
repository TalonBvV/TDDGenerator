"""
Global augmentation module for applying photometric and geometric transforms.
"""

import logging
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .ingest import Polygon

logger = logging.getLogger("engine.augment")


class Augmentor:
    """Applies global augmentations to composed images."""
    
    def __init__(self, probability: float = 0.5):
        """
        Initialize augmentor.
        
        Args:
            probability: Probability of applying augmentations (0.0 to 1.0)
        """
        self.probability = probability
    
    def augment(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """
        Apply random augmentations to image and polygons.
        
        Args:
            image: RGB numpy array
            polygons: List of polygon annotations
            
        Returns:
            Tuple of (augmented image, transformed polygons)
        """
        if random.random() > self.probability:
            return image, polygons
        
        aug_funcs = [
            self._brightness_contrast,
            self._exposure,
            self._color_temperature,
            self._jpeg_artifacts,
            self._gaussian_noise,
            self._motion_blur,
            self._lens_distortion,
            self._fog,
            self._rain,
            self._paper_texture,
            self._hdr_effect,
            self._lens_flare,
            self._bleed_through,
            self._rolling_shutter,
        ]
        
        num_augs = random.randint(1, 3)
        selected_augs = random.sample(aug_funcs, min(num_augs, len(aug_funcs)))
        
        result_image = image.copy()
        result_polygons = polygons
        
        for aug_func in selected_augs:
            try:
                result_image, result_polygons = aug_func(result_image, result_polygons)
            except Exception as e:
                logger.warning(f"Augmentation {aug_func.__name__} failed: {e}")
        
        return result_image, result_polygons
    
    def _brightness_contrast(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Adjust brightness and contrast."""
        brightness = random.uniform(-30, 30)
        contrast = random.uniform(0.8, 1.2)
        
        result = image.astype(np.float32)
        result = contrast * (result - 127.5) + 127.5 + brightness
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, polygons
    
    def _exposure(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Adjust exposure."""
        ev = random.uniform(-0.5, 0.5)
        factor = 2 ** ev
        
        result = image.astype(np.float32) * factor
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, polygons
    
    def _color_temperature(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Adjust color temperature."""
        temp = random.uniform(-0.15, 0.15)
        
        result = image.astype(np.float32)
        
        if temp > 0:
            result[:, :, 0] = np.clip(result[:, :, 0] * (1 + temp), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 - temp * 0.5), 0, 255)
        else:
            result[:, :, 0] = np.clip(result[:, :, 0] * (1 + temp * 0.5), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 - temp), 0, 255)
        
        return result.astype(np.uint8), polygons
    
    def _jpeg_artifacts(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add JPEG compression artifacts."""
        quality = random.randint(50, 90)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        
        return result, polygons
    
    def _gaussian_noise(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add Gaussian noise."""
        sigma = random.uniform(5, 20)
        
        noise = np.random.normal(0, sigma, image.shape)
        result = image.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, polygons
    
    def _motion_blur(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add motion blur."""
        kernel_size = random.choice([3, 5, 7])
        angle = random.uniform(0, 180)
        
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(
            kernel,
            rotation_matrix,
            (kernel_size, kernel_size)
        )
        
        kernel = kernel / kernel.sum()
        
        result = cv2.filter2D(image, -1, kernel)
        
        return result, polygons
    
    def _lens_distortion(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Apply barrel/pincushion lens distortion with accurate polygon transform."""
        h, w = image.shape[:2]
        k1 = random.uniform(-0.08, 0.08)
        
        fx = fy = max(h, w) * 1.2
        cx, cy = w / 2, h / 2
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.array([k1, k1 * 0.1, 0, 0, 0], dtype=np.float32)
        
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0
        )
        
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )
        
        result = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        
        new_polygons = []
        for polygon in polygons:
            new_points = []
            for px, py in polygon.points:
                x = px * w
                y = py * h
                
                x_norm = (x - cx) / fx
                y_norm = (y - cy) / fy
                r2 = x_norm ** 2 + y_norm ** 2
                
                radial = 1 + k1 * r2 + (k1 * 0.1) * r2 * r2
                
                x_dist = x_norm * radial
                y_dist = y_norm * radial
                
                new_x = (x_dist * new_camera_matrix[0, 0] + new_camera_matrix[0, 2]) / w
                new_y = (y_dist * new_camera_matrix[1, 1] + new_camera_matrix[1, 2]) / h
                
                new_x = max(0, min(1, new_x))
                new_y = max(0, min(1, new_y))
                
                new_points.append((new_x, new_y))
            
            new_polygons.append(Polygon(class_id=polygon.class_id, points=new_points))
        
        return result, new_polygons
    
    def _fog(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add fog effect."""
        density = random.uniform(0.1, 0.3)
        
        h, w = image.shape[:2]
        fog = np.ones_like(image, dtype=np.float32) * 220
        
        noise = np.random.normal(0, 20, (h, w))
        noise = cv2.GaussianBlur(noise, (0, 0), 50)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        
        fog_mask = (density + noise[:, :, np.newaxis] * 0.2).clip(0, 1)
        
        result = image.astype(np.float32) * (1 - fog_mask) + fog * fog_mask
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, polygons
    
    def _rain(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add rain streaks."""
        h, w = image.shape[:2]
        
        rain_layer = np.zeros((h, w), dtype=np.float32)
        
        num_drops = random.randint(100, 300)
        drop_length = random.randint(10, 30)
        drop_width = 1
        angle = random.uniform(-10, 10)
        
        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            dx = int(drop_length * np.sin(np.radians(angle)))
            dy = drop_length
            
            cv2.line(
                rain_layer,
                (x, y),
                (x + dx, y + dy),
                random.uniform(0.3, 0.7),
                drop_width
            )
        
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        
        rain_color = np.array([200, 200, 220], dtype=np.float32)
        rain_rgb = rain_layer[:, :, np.newaxis] * rain_color
        
        result = image.astype(np.float32) + rain_rgb * 0.5
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, polygons
    
    def _paper_texture(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add paper/document texture effects."""
        effect_type = random.choice(["stain", "fold", "crumple"])
        
        if effect_type == "stain":
            return self._add_stain(image, polygons)
        elif effect_type == "fold":
            return self._add_fold(image, polygons)
        else:
            return self._add_crumple(image, polygons)
    
    def _add_stain(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add coffee/water stain."""
        h, w = image.shape[:2]
        
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        radius = random.randint(30, 100)
        
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        
        mask = np.clip(1 - dist / radius, 0, 1) ** 2
        
        noise = np.random.normal(0, 0.3, (h, w))
        noise = cv2.GaussianBlur(noise, (21, 21), 0)
        mask = np.clip(mask + noise * mask, 0, 1)
        
        stain_color = np.array([
            random.randint(100, 150),
            random.randint(80, 120),
            random.randint(50, 80)
        ], dtype=np.float32)
        
        stain = mask[:, :, np.newaxis] * stain_color
        
        result = image.astype(np.float32)
        result = result * (1 - mask[:, :, np.newaxis] * 0.3) + stain * 0.3
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, polygons
    
    def _add_fold(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add paper fold line."""
        h, w = image.shape[:2]
        
        if random.random() < 0.5:
            x = random.randint(w // 4, 3 * w // 4)
            mask = np.abs(np.arange(w) - x) / (w * 0.1)
            mask = np.clip(1 - mask, 0, 1)
            mask = mask[np.newaxis, :, np.newaxis]
        else:
            y = random.randint(h // 4, 3 * h // 4)
            mask = np.abs(np.arange(h) - y) / (h * 0.1)
            mask = np.clip(1 - mask, 0, 1)
            mask = mask[:, np.newaxis, np.newaxis]
        
        shadow = -20 * mask
        highlight = 10 * np.roll(mask, 5, axis=1 if mask.shape[1] > 1 else 0)
        
        result = image.astype(np.float32) + shadow + highlight
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, polygons
    
    def _add_crumple(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add crumple/wrinkle texture."""
        h, w = image.shape[:2]
        
        noise = np.random.normal(0, 1, (h // 8, w // 8))
        noise = cv2.resize(noise, (w, h))
        noise = cv2.GaussianBlur(noise, (0, 0), 3)
        
        gx = cv2.Sobel(noise, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(noise, cv2.CV_32F, 0, 1, ksize=3)
        
        shading = (gx + gy) * 15
        
        result = image.astype(np.float32) + shading[:, :, np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, polygons
    
    def _hdr_effect(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Apply HDR-like tone mapping effect."""
        strength = random.uniform(0.3, 0.7)
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel.astype(np.uint8))
        
        lab[:, :, 0] = l_channel * (1 - strength) + l_enhanced * strength
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + strength * 0.3), 0, 255)
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return result, polygons
    
    def _lens_flare(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add lens flare effect."""
        h, w = image.shape[:2]
        
        fx = random.randint(0, w)
        fy = random.randint(0, h // 3)
        
        result = image.astype(np.float32)
        
        y_grid, x_grid = np.ogrid[:h, :w]
        
        for i in range(random.randint(3, 6)):
            cx = int(fx + (w // 2 - fx) * i * 0.2)
            cy = int(fy + (h // 2 - fy) * i * 0.2)
            radius = random.randint(20, 80)
            
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            flare = np.clip(1 - dist / radius, 0, 1) ** 2
            
            color = np.array([
                random.randint(200, 255),
                random.randint(180, 240),
                random.randint(150, 220)
            ], dtype=np.float32)
            
            intensity = random.uniform(0.1, 0.4)
            result += flare[:, :, np.newaxis] * color * intensity
        
        main_dist = np.sqrt((x_grid - fx) ** 2 + (y_grid - fy) ** 2)
        main_flare = np.clip(1 - main_dist / 150, 0, 1) ** 1.5
        result += main_flare[:, :, np.newaxis] * np.array([255, 240, 200]) * 0.5
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result, polygons
    
    def _bleed_through(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Add document bleed-through effect (text from back of page)."""
        h, w = image.shape[:2]
        
        ghost = np.zeros((h, w), dtype=np.float32)
        
        num_lines = random.randint(3, 8)
        for _ in range(num_lines):
            y = random.randint(h // 4, 3 * h // 4)
            x1 = random.randint(w // 6, w // 3)
            x2 = random.randint(2 * w // 3, 5 * w // 6)
            thickness = random.randint(2, 5)
            
            cv2.line(ghost, (x1, y), (x2, y), random.uniform(0.3, 0.6), thickness)
        
        ghost = cv2.flip(ghost, 1)
        
        ghost = cv2.GaussianBlur(ghost, (7, 7), 2)
        
        intensity = random.uniform(0.05, 0.15)
        ghost_color = np.array([80, 80, 100], dtype=np.float32)
        
        result = image.astype(np.float32)
        result = result * (1 - ghost[:, :, np.newaxis] * intensity) + \
                 ghost[:, :, np.newaxis] * ghost_color * intensity
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result, polygons
    
    def _rolling_shutter(
        self,
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]:
        """Apply rolling shutter/skew effect."""
        h, w = image.shape[:2]
        
        skew = random.uniform(-0.05, 0.05) * w
        
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        offset = (yy / h) * skew
        x_map = (xx + offset).astype(np.float32)
        y_map = yy.astype(np.float32)
        
        result = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        
        new_polygons = []
        for polygon in polygons:
            new_points = []
            for px, py in polygon.points:
                x = px * w
                y = py * h
                
                new_x = x + (y / h) * skew
                new_x = max(0, min(w - 1, new_x))
                
                new_points.append((new_x / w, py))
            
            new_polygons.append(Polygon(class_id=polygon.class_id, points=new_points))
        
        return result, new_polygons
