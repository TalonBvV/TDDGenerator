"""
Synthetic text generation module for rendering text with various effects.
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .extract_assets import WordAsset

logger = logging.getLogger("engine.synth_generator")


class TextEffect(Enum):
    """Available text effects."""
    NONE = "none"
    STROKE = "stroke"
    SHADOW = "shadow"
    GRADIENT = "gradient"
    EMBOSS = "emboss"


@dataclass
class StrokeConfig:
    """Configuration for text stroke/outline."""
    width: int
    color: Tuple[int, int, int]


@dataclass
class ShadowConfig:
    """Configuration for drop shadow."""
    offset: Tuple[int, int]
    blur: float
    opacity: float
    color: Tuple[int, int, int]


@dataclass
class GradientConfig:
    """Configuration for gradient fill."""
    color1: Tuple[int, int, int]
    color2: Tuple[int, int, int]
    direction: str


class TextRenderer:
    """Renders text with various styles and effects."""
    
    def __init__(self, fonts_dir: Path):
        self.fonts_dir = Path(fonts_dir)
        self.fonts: List[Path] = []
        self._load_fonts()
    
    def _load_fonts(self) -> None:
        """Load available fonts from the fonts directory."""
        if not self.fonts_dir.exists():
            logger.warning(f"Fonts directory not found: {self.fonts_dir}")
            return
        
        for ext in ["*.ttf", "*.otf", "*.TTF", "*.OTF"]:
            self.fonts.extend(self.fonts_dir.glob(ext))
        
        if not self.fonts:
            logger.warning("No fonts found in fonts directory")
        else:
            logger.info(f"Loaded {len(self.fonts)} fonts")
    
    def get_random_font(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        """Get a random font at the specified size."""
        if not self.fonts:
            return ImageFont.load_default()
        
        font_path = random.choice(self.fonts)
        
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception as e:
            logger.warning(f"Failed to load font {font_path}: {e}")
            return ImageFont.load_default()
    
    def render_word(
        self,
        word: str,
        font_size: int,
        color: Tuple[int, int, int],
        stroke: Optional[StrokeConfig] = None,
        shadow: Optional[ShadowConfig] = None,
        gradient: Optional[GradientConfig] = None,
        emboss: bool = False
    ) -> np.ndarray:
        """
        Render a word with the specified style.
        
        Returns:
            RGBA numpy array
        """
        font = self.get_random_font(font_size)
        
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        bbox = dummy_draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = max(20, font_size // 2)
        if stroke:
            padding += stroke.width * 2
        if shadow:
            padding += max(abs(shadow.offset[0]), abs(shadow.offset[1])) + int(shadow.blur * 2)
        
        img_width = text_width + padding * 2
        img_height = text_height + padding * 2
        
        image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        text_x = padding - bbox[0]
        text_y = padding - bbox[1]
        
        if shadow:
            shadow_layer = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            
            shadow_x = text_x + shadow.offset[0]
            shadow_y = text_y + shadow.offset[1]
            
            shadow_color = (*shadow.color, int(255 * shadow.opacity))
            shadow_draw.text((shadow_x, shadow_y), word, font=font, fill=shadow_color)
            
            if shadow.blur > 0:
                shadow_layer = shadow_layer.filter(
                    ImageFilter.GaussianBlur(radius=shadow.blur)
                )
            
            image = Image.alpha_composite(image, shadow_layer)
            draw = ImageDraw.Draw(image)
        
        if stroke:
            stroke_color = (*stroke.color, 255)
            draw.text(
                (text_x, text_y),
                word,
                font=font,
                fill=(*color, 255),
                stroke_width=stroke.width,
                stroke_fill=stroke_color
            )
        else:
            draw.text((text_x, text_y), word, font=font, fill=(*color, 255))
        
        if gradient:
            image = self._apply_gradient(image, gradient, (text_x, text_y, text_width, text_height))
        
        if emboss:
            image = self._apply_emboss(image)
        
        image = self._trim_transparent(image)
        
        return np.array(image)
    
    def _apply_gradient(
        self,
        image: Image.Image,
        gradient: GradientConfig,
        text_region: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Apply gradient to the text."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        if gradient.direction == "horizontal":
            grad = np.linspace(0, 1, w).reshape(1, w, 1)
        else:
            grad = np.linspace(0, 1, h).reshape(h, 1, 1)
        
        color1 = np.array(gradient.color1).reshape(1, 1, 3)
        color2 = np.array(gradient.color2).reshape(1, 1, 3)
        
        grad_colors = (1 - grad) * color1 + grad * color2
        grad_colors = np.broadcast_to(grad_colors, (h, w, 3)).astype(np.uint8)
        
        alpha = img_array[:, :, 3:4] / 255.0
        img_array[:, :, :3] = (alpha * grad_colors + (1 - alpha) * img_array[:, :, :3]).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _apply_emboss(self, image: Image.Image) -> Image.Image:
        """Apply emboss effect to the image."""
        embossed = image.filter(ImageFilter.EMBOSS)
        
        result = Image.blend(image, embossed, alpha=0.3)
        
        result.putalpha(image.split()[3])
        
        return result
    
    def _trim_transparent(self, image: Image.Image) -> Image.Image:
        """Trim transparent borders from image."""
        img_array = np.array(image)
        alpha = img_array[:, :, 3]
        
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        
        if not rows.any() or not cols.any():
            return image
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        padding = 2
        y_min = max(0, y_min - padding)
        y_max = min(img_array.shape[0], y_max + padding + 1)
        x_min = max(0, x_min - padding)
        x_max = min(img_array.shape[1], x_max + padding + 1)
        
        return Image.fromarray(img_array[y_min:y_max, x_min:x_max])


class SynthGenerator:
    """Generates synthetic text word assets."""
    
    def __init__(
        self,
        fonts_dir: Path,
        wordlist: List[str],
        output_dir: Path
    ):
        self.fonts_dir = Path(fonts_dir)
        self.wordlist = wordlist
        self.output_dir = Path(output_dir)
        self.renderer = TextRenderer(fonts_dir)
        
        self.synth_text_dir = self.output_dir / "assets" / "synth_text"
        self.synth_text_dir.mkdir(parents=True, exist_ok=True)
    
    def _random_color(self) -> Tuple[int, int, int]:
        """Generate a random color."""
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    
    def _random_stroke(self) -> Optional[StrokeConfig]:
        """Generate random stroke config (50% chance)."""
        if random.random() < 0.5:
            return None
        
        return StrokeConfig(
            width=random.randint(1, 3),
            color=self._random_color()
        )
    
    def _random_shadow(self) -> Optional[ShadowConfig]:
        """Generate random shadow config (50% chance)."""
        if random.random() < 0.5:
            return None
        
        offset_x = random.randint(2, 5) * random.choice([-1, 1])
        offset_y = random.randint(2, 5)
        
        return ShadowConfig(
            offset=(offset_x, offset_y),
            blur=random.uniform(1, 3),
            opacity=random.uniform(0.3, 0.6),
            color=(0, 0, 0)
        )
    
    def _random_gradient(self) -> Optional[GradientConfig]:
        """Generate random gradient config (30% chance)."""
        if random.random() < 0.7:
            return None
        
        return GradientConfig(
            color1=self._random_color(),
            color2=self._random_color(),
            direction=random.choice(["horizontal", "vertical"])
        )
    
    def generate_single(
        self,
        word: Optional[str] = None,
        font_size: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Generate a single synthetic word asset."""
        if word is None:
            word = random.choice(self.wordlist)
        
        if font_size is None:
            font_size = random.randint(24, 72)
        
        if color is None:
            color = self._random_color()
        
        stroke = self._random_stroke()
        shadow = self._random_shadow()
        gradient = self._random_gradient()
        emboss = random.random() < 0.2
        
        return self.renderer.render_word(
            word=word,
            font_size=font_size,
            color=color,
            stroke=stroke,
            shadow=shadow,
            gradient=gradient,
            emboss=emboss
        )
    
    def generate(
        self,
        count: int,
        target_dataset: str,
        save_assets: bool = True
    ) -> List[WordAsset]:
        """Generate multiple synthetic word assets."""
        logger.info(f"Generating {count} synthetic assets for dataset '{target_dataset}'")
        
        dataset_dir = self.synth_text_dir / target_dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        assets = []
        
        for i in range(count):
            word = random.choice(self.wordlist)
            image = self.generate_single(word=word)
            
            asset = WordAsset(
                image=image,
                source_dataset=target_dataset,
                source_image=f"synth_{i:06d}",
                original_polygon=None,
                bbox=(0, 0, image.shape[1], image.shape[0])
            )
            
            if save_assets:
                asset_path = dataset_dir / f"synth_{i:06d}.png"
                Image.fromarray(image).save(str(asset_path))
                asset.asset_path = asset_path
            
            assets.append(asset)
        
        logger.info(f"Generated {len(assets)} synthetic assets")
        
        return assets
    
    def generate_for_background(
        self,
        background_region: np.ndarray,
        word: Optional[str] = None,
        target_height: Optional[int] = None
    ) -> np.ndarray:
        """Generate a synthetic word with color contrasting the background."""
        from ..utils import get_contrasting_color
        
        if word is None:
            word = random.choice(self.wordlist)
        
        color = get_contrasting_color(background_region)
        
        if target_height:
            font_size = int(target_height * 0.8)
        else:
            font_size = random.randint(24, 72)
        
        return self.renderer.render_word(
            word=word,
            font_size=font_size,
            color=color,
            stroke=self._random_stroke(),
            shadow=self._random_shadow(),
            gradient=self._random_gradient(),
            emboss=random.random() < 0.2
        )
