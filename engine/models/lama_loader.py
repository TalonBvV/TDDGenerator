"""
LaMa model loader for image inpainting.
Downloads big-lama from HuggingFace if not present locally.
"""

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger("engine.models.lama")

LAMA_HF_URL = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"


def get_cache_dir() -> Path:
    """Get the cache directory for model storage."""
    cache_dir = Path.home() / ".cache" / "text_detection_engine" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    print(f"\rDownloading big-lama: {percent}% ({downloaded // (1024*1024)}MB)", end="", flush=True)


def download_lama_model(target_dir: Path) -> Path:
    """
    Download big-lama model from HuggingFace.
    
    Args:
        target_dir: Directory to extract model to
        
    Returns:
        Path to the extracted model directory
    """
    model_dir = target_dir / "big-lama"
    
    if model_dir.exists() and (model_dir / "models" / "best.ckpt").exists():
        logger.info(f"LaMa model already exists at {model_dir}")
        return model_dir
    
    zip_path = target_dir / "big-lama.zip"
    
    logger.info(f"Downloading big-lama from HuggingFace...")
    logger.info(f"URL: {LAMA_HF_URL}")
    
    try:
        urlretrieve(LAMA_HF_URL, zip_path, reporthook=download_progress)
        print()
    except Exception as e:
        logger.error(f"Failed to download via urlretrieve: {e}")
        logger.info("Attempting download via curl...")
        
        try:
            subprocess.run(
                ["curl", "-LJO", LAMA_HF_URL],
                cwd=str(target_dir),
                check=True,
                capture_output=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as curl_err:
            raise RuntimeError(
                f"Failed to download LaMa model. Please download manually:\n"
                f"  curl -LJO {LAMA_HF_URL}\n"
                f"  unzip big-lama.zip -d {target_dir}"
            ) from curl_err
    
    logger.info(f"Extracting model to {target_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    
    zip_path.unlink()
    
    logger.info(f"LaMa model extracted to {model_dir}")
    return model_dir


def load_lama_checkpoint(model_path: Path, device: str) -> nn.Module:
    """
    Load LaMa model from checkpoint.
    
    Args:
        model_path: Path to big-lama directory
        device: Device to load model on
        
    Returns:
        Loaded LaMa model
    """
    sys.path.insert(0, str(model_path))
    
    try:
        from saicinpainting.training.trainers import load_checkpoint
        from saicinpainting.evaluation.utils import move_to_device
        from omegaconf import OmegaConf
        
        config_path = model_path / "config.yaml"
        checkpoint_path = model_path / "models" / "best.ckpt"
        
        with open(config_path, 'r') as f:
            predict_config = OmegaConf.create(yaml.safe_load(f))
        
        predict_config.model.path = str(checkpoint_path)
        
        model = load_checkpoint(
            predict_config,
            str(checkpoint_path),
            strict=False,
            map_location=device
        )
        model = model.to(device)
        model.eval()
        
        return model
        
    except ImportError:
        logger.warning("LaMa dependencies not in path, using direct loading...")
        return load_lama_direct(model_path, device)
    finally:
        if str(model_path) in sys.path:
            sys.path.remove(str(model_path))


def load_lama_direct(model_path: Path, device: str) -> "LamaInferenceModel":
    """
    Direct loading of LaMa model without full saicinpainting dependency.
    """
    checkpoint_path = model_path / "models" / "best.ckpt"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    return LamaInferenceModel(checkpoint, device)


class LamaInferenceModel(nn.Module):
    """Simplified LaMa inference wrapper."""
    
    def __init__(self, checkpoint: dict, device: str):
        super().__init__()
        self.device = device
        self.checkpoint = checkpoint
        self._model = None
        self._build_model()
    
    def _build_model(self):
        """Build model from checkpoint."""
        try:
            from saicinpainting.training.modules import make_generator
            
            state_dict = self.checkpoint.get('state_dict', self.checkpoint)
            
            generator_state = {
                k.replace('generator.', ''): v 
                for k, v in state_dict.items() 
                if k.startswith('generator.')
            }
            
            self._model = make_generator(
                kind='ffc_resnet', 
                input_nc=4, 
                output_nc=3,
                ngf=64,
                n_downsampling=3,
                n_blocks=18
            )
            self._model.load_state_dict(generator_state, strict=False)
            self._model = self._model.to(self.device)
            self._model.eval()
            
        except Exception as e:
            logger.warning(f"Could not build full model: {e}")
            self._model = None
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run inference.
        
        Args:
            image: (B, 3, H, W) normalized to [0, 1]
            mask: (B, 1, H, W) binary mask
            
        Returns:
            Inpainted image (B, 3, H, W)
        """
        if self._model is None:
            raise RuntimeError("Model not properly loaded")
        
        masked_image = image * (1 - mask)
        
        input_tensor = torch.cat([masked_image, mask], dim=1)
        
        with torch.no_grad():
            output = self._model(input_tensor)
        
        result = image * (1 - mask) + output * mask
        
        return result


class LamaModel:
    """Wrapper for LaMa (Large Mask Inpainting) model."""
    
    def __init__(
        self,
        checkpoint: str = "big-lama",
        device: str = "cuda",
        cache_dir: Optional[Path] = None
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.cache_dir = cache_dir or get_cache_dir()
        self.model = None
        self.model_path: Optional[Path] = None
        self._loaded = False
        self._use_simple_lama = False
    
    def load(self) -> None:
        """Load the LaMa model."""
        if self._loaded:
            return
        
        logger.info(f"Loading LaMa model: {self.checkpoint}")
        
        try:
            from simple_lama_inpainting import SimpleLama
            self.model = SimpleLama()
            self._loaded = True
            self._use_simple_lama = True
            logger.info("LaMa model loaded via simple-lama-inpainting")
            return
        except ImportError:
            logger.info("simple-lama-inpainting not available, using direct loading...")
        
        self.model_path = download_lama_model(self.cache_dir)
        
        try:
            self.model = load_lama_checkpoint(self.model_path, self.device)
            self._loaded = True
            logger.info("LaMa model loaded successfully from checkpoint")
        except Exception as e:
            logger.error(f"Failed to load LaMa checkpoint: {e}")
            logger.warning("Falling back to OpenCV inpainting")
            self._loaded = True
            self._use_fallback = True
    
    def _preprocess(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pad_to_modulo: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Preprocess image and mask for LaMa."""
        h, w = image.shape[:2]
        
        pad_h = (pad_to_modulo - h % pad_to_modulo) % pad_to_modulo
        pad_w = (pad_to_modulo - w % pad_to_modulo) % pad_to_modulo
        
        img = image.astype(np.float32) / 255.0
        
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        msk = (mask > 127).astype(np.float32)
        
        if pad_h > 0 or pad_w > 0:
            msk = np.pad(msk, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        msk = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0)
        
        return img.to(self.device), msk.to(self.device), (h, w)
    
    def _postprocess(self, output: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess LaMa output."""
        h, w = original_size
        result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = result[:h, :w]
        result = (result * 255).clip(0, 255).astype(np.uint8)
        return result
    
    def inpaint_single(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Inpaint a single image.
        
        Args:
            image: RGB numpy array (H, W, 3)
            mask: Binary mask where white (255) indicates regions to inpaint
            
        Returns:
            Inpainted RGB numpy array
        """
        if not self._loaded:
            self.load()
        
        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._opencv_inpaint(image, mask)
        
        if self._use_simple_lama:
            from PIL import Image as PILImage
            
            pil_image = PILImage.fromarray(image)
            
            if mask.ndim == 3:
                mask_2d = mask[:, :, 0]
            else:
                mask_2d = mask
            pil_mask = PILImage.fromarray(mask_2d)
            
            result = self.model(pil_image, pil_mask)
            return np.array(result)
        
        img_t, mask_t, orig_size = self._preprocess(image, mask)
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                output = self.model(img_t, mask_t)
            else:
                output = self.model(img_t, mask_t)
        
        return self._postprocess(output, orig_size)
    
    def _opencv_inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Fallback inpainting using OpenCV."""
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        mask_binary = (mask > 127).astype(np.uint8) * 255
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        result = cv2.inpaint(
            image_bgr,
            mask_binary,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    @torch.no_grad()
    def inpaint_batch(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Inpaint a batch of images.
        
        Args:
            images: List of RGB numpy arrays
            masks: List of binary masks
            
        Returns:
            List of inpainted RGB numpy arrays
        """
        if not self._loaded:
            self.load()
        
        if hasattr(self, "_use_fallback") and self._use_fallback:
            return [self._opencv_inpaint(img, msk) for img, msk in zip(images, masks)]
        
        if self._use_simple_lama:
            return [self.inpaint_single(img, msk) for img, msk in zip(images, masks)]
        
        if len(images) == 0:
            return []
        
        all_same_size = all(
            img.shape[:2] == images[0].shape[:2] 
            for img in images
        )
        
        if not all_same_size:
            return [self.inpaint_single(img, msk) for img, msk in zip(images, masks)]
        
        batch_imgs = []
        batch_masks = []
        orig_size = None
        
        for image, mask in zip(images, masks):
            img_t, mask_t, orig_size = self._preprocess(image, mask)
            batch_imgs.append(img_t)
            batch_masks.append(mask_t)
        
        batch_img = torch.cat(batch_imgs, dim=0)
        batch_mask = torch.cat(batch_masks, dim=0)
        
        with torch.no_grad():
            output = self.model(batch_img, batch_mask)
        
        results = []
        for i in range(output.shape[0]):
            result = self._postprocess(output[i:i+1], orig_size)
            results.append(result)
        
        return results
    
    def unload(self) -> None:
        """Unload the model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self._loaded = False
        self._use_simple_lama = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("LaMa model unloaded")
