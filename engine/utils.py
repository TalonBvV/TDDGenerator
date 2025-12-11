"""
Common utilities for the dataset generator engine.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def setup_logging(output_dir: Path, verbose: bool = True) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"generation_{timestamp}.log"
    
    logger = logging.getLogger("engine")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_rgba(path: Union[str, Path]) -> np.ndarray:
    """Load image as RGBA numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    
    return img


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """Save RGB/RGBA numpy array as image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    cv2.imwrite(str(path), img)


def save_image_pil(img: np.ndarray, path: Union[str, Path]) -> None:
    """Save image using PIL (better for RGBA PNGs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(img).save(str(path))


def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect: bool = False
) -> np.ndarray:
    """Resize image to target size."""
    if keep_aspect:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        if img.ndim == 3:
            canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=img.dtype)
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    else:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)


def crop_to_square(img: np.ndarray) -> np.ndarray:
    """Center crop image to square."""
    h, w = img.shape[:2]
    size = min(h, w)
    
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    
    return img[y_start:y_start + size, x_start:x_start + size]


def polygon_to_mask(
    polygon: List[Tuple[float, float]],
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Convert normalized polygon coordinates to binary mask."""
    h, w = image_size
    
    points = np.array([
        [int(x * w), int(y * h)] for x, y in polygon
    ], dtype=np.int32)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    return mask


def mask_to_polygon(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Convert binary mask to normalized polygon coordinates."""
    h, w = mask.shape[:2]
    
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    polygon = [(p[0][0] / w, p[0][1] / h) for p in approx]
    
    return polygon


def get_bbox_from_polygon(
    polygon: List[Tuple[float, float]],
    image_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Get bounding box from polygon coordinates."""
    h, w = image_size
    
    xs = [int(x * w) for x, y in polygon]
    ys = [int(y * h) for x, y in polygon]
    
    return min(xs), min(ys), max(xs), max(ys)


def crop_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """Crop image region with alpha from mask."""
    x1, y1, x2, y2 = bbox
    
    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2].copy()
    
    if cropped_img.shape[2] == 3:
        rgba = np.zeros((*cropped_img.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = cropped_img
        rgba[:, :, 3] = cropped_mask
        return rgba
    else:
        cropped_img[:, :, 3] = cropped_mask
        return cropped_img


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def get_contrasting_color(
    background_region: np.ndarray,
    min_contrast: float = 50.0
) -> Tuple[int, int, int]:
    """Get a color that contrasts with the background region."""
    if background_region.size == 0:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    mean_color = background_region.mean(axis=(0, 1))
    
    if len(mean_color) >= 3:
        mean_l = 0.299 * mean_color[0] + 0.587 * mean_color[1] + 0.114 * mean_color[2]
    else:
        mean_l = mean_color[0]
    
    for _ in range(100):
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        color_l = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        
        if abs(color_l - mean_l) >= min_contrast:
            return color
    
    if mean_l > 127:
        return (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    else:
        return (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))


def apply_color_transfer(
    source: np.ndarray,
    target: np.ndarray,
    preserve_luminance: bool = True,
    strength: float = 0.7
) -> np.ndarray:
    """
    Apply color transfer from target to source using LAB color space.
    Improved handling for edge cases and text readability.
    
    Args:
        source: Source image (text asset)
        target: Target image (background region)
        preserve_luminance: Keep original luminance to preserve text readability
        strength: Blending strength (0-1), lower = more subtle
    """
    if source.size == 0 or target.size == 0:
        return source
    
    source_rgb = source[:, :, :3] if source.shape[2] >= 3 else source
    target_rgb = target[:, :, :3] if target.shape[2] >= 3 else target
    
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    src_mean = source_lab.mean(axis=(0, 1))
    src_std = source_lab.std(axis=(0, 1))
    tgt_mean = target_lab.mean(axis=(0, 1))
    tgt_std = target_lab.std(axis=(0, 1))
    
    src_std = np.maximum(src_std, 1.0)
    tgt_std = np.maximum(tgt_std, 1.0)
    
    scale = np.clip(tgt_std / src_std, 0.5, 2.0)
    
    result_lab = source_lab.copy()
    
    if preserve_luminance:
        for c in [1, 2]:
            result_lab[:, :, c] = (source_lab[:, :, c] - src_mean[c]) * scale[c] + tgt_mean[c]
    else:
        for c in [0, 1, 2]:
            result_lab[:, :, c] = (source_lab[:, :, c] - src_mean[c]) * scale[c] + tgt_mean[c]
    
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)
    
    result_lab = result_lab.astype(np.uint8)
    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    
    if strength < 1.0:
        result_rgb = (source_rgb * (1 - strength) + result_rgb * strength).astype(np.uint8)
    
    if source.shape[2] == 4:
        result = np.zeros_like(source)
        result[:, :, :3] = result_rgb
        result[:, :, 3] = source[:, :, 3]
        return result
    
    return result_rgb


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """Save generation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def weighted_random_choice(items: List[Any], weights: List[float]) -> Any:
    """Select an item with weighted probability."""
    total = sum(weights)
    weights = [w / total for w in weights]
    return random.choices(items, weights=weights, k=1)[0]


def batch_iterator(items: List[Any], batch_size: int):
    """Yield batches of items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
