"""
RMBG-2.0 model loader for background removal / alpha matting.
Robust output handling for different transformers versions.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("engine.models.rmbg")


def extract_mask_from_output(outputs, index: int = 0) -> torch.Tensor:
    """
    Extract mask tensor from model outputs, handling various output formats.
    
    Args:
        outputs: Model output (can be dict, tuple, tensor, or object with attributes)
        index: Batch index to extract
        
    Returns:
        2D mask tensor
    """
    mask = None
    
    if hasattr(outputs, 'preds'):
        mask = outputs.preds
    elif hasattr(outputs, 'logits'):
        mask = outputs.logits
    elif hasattr(outputs, 'pred_masks'):
        mask = outputs.pred_masks
    elif hasattr(outputs, 'masks'):
        mask = outputs.masks
    elif isinstance(outputs, dict):
        for key in ['preds', 'logits', 'pred_masks', 'masks', 'output']:
            if key in outputs:
                mask = outputs[key]
                break
        if mask is None and len(outputs) > 0:
            mask = list(outputs.values())[0]
    elif isinstance(outputs, (tuple, list)):
        mask = outputs[0]
    elif isinstance(outputs, torch.Tensor):
        mask = outputs
    else:
        raise ValueError(f"Unknown output type: {type(outputs)}")
    
    if mask is None:
        raise ValueError("Could not extract mask from model output")
    
    if isinstance(mask, (list, tuple)):
        mask = mask[index] if len(mask) > index else mask[0]
    elif mask.dim() >= 1 and mask.shape[0] > index:
        mask = mask[index]
    
    while mask.dim() > 2:
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            mask = mask[0]
    
    if mask.dim() == 1:
        size = int(np.sqrt(mask.shape[0]))
        mask = mask.view(size, size)
    
    return mask


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Normalize mask values to [0, 1] range."""
    if mask.dtype == torch.bool:
        return mask.float()
    
    min_val = mask.min()
    max_val = mask.max()
    
    if max_val - min_val < 1e-6:
        return torch.zeros_like(mask)
    
    if min_val < 0 or max_val > 1:
        mask = torch.sigmoid(mask)
    
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    
    return mask


class RMBGModel:
    """Wrapper for BRIA RMBG-2.0 background removal model."""
    
    def __init__(
        self,
        model_name: str = "briaai/RMBG-2.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self._loaded = False
        self._output_format = None
    
    def load(self) -> None:
        """Load the RMBG model."""
        if self._loaded:
            return
        
        logger.info(f"Loading RMBG model: {self.model_name}")
        
        try:
            from transformers import AutoModelForImageSegmentation
            
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            except Exception:
                from transformers import AutoImageProcessor
                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = self.model.to(self.device)
            
            if self.device == "cuda" and self.dtype == torch.float16:
                self.model = self.model.half()
            
            self.model.eval()
            self._loaded = True
            
            logger.info("RMBG model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load RMBG model: {e}")
            raise
    
    def _process_inputs(self, pil_images: Union[Image.Image, List[Image.Image]]) -> dict:
        """Process input images through the processor."""
        try:
            inputs = self.processor(images=pil_images, return_tensors="pt")
        except TypeError:
            inputs = self.processor(pil_images, return_tensors="pt")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.dtype == torch.float16:
            inputs = {
                k: v.half() if v.dtype == torch.float32 else v 
                for k, v in inputs.items()
            }
        
        return inputs
    
    def _extract_alpha(
        self,
        outputs,
        original_size: tuple,
        batch_index: int = 0
    ) -> np.ndarray:
        """Extract alpha channel from model outputs."""
        mask = extract_mask_from_output(outputs, batch_index)
        
        mask = normalize_mask(mask)
        
        if mask.shape[0] != original_size[1] or mask.shape[1] != original_size[0]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False
            ).squeeze()
        
        alpha = (mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        return alpha
    
    def process_single(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image to extract alpha matte.
        
        Args:
            image: RGB numpy array (H, W, 3)
            
        Returns:
            RGBA numpy array (H, W, 4) with alpha matte
        """
        if not self._loaded:
            self.load()
        
        pil_image = Image.fromarray(image)
        original_size = pil_image.size
        
        inputs = self._process_inputs(pil_image)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        alpha = self._extract_alpha(outputs, original_size, batch_index=0)
        
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = alpha
        
        return rgba
    
    @torch.no_grad()
    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of images.
        
        Args:
            images: List of RGB numpy arrays
            
        Returns:
            List of RGBA numpy arrays with alpha mattes
        """
        if not self._loaded:
            self.load()
        
        if not images:
            return []
        
        pil_images = [Image.fromarray(img) for img in images]
        original_sizes = [img.size for img in pil_images]
        
        try:
            inputs = self._process_inputs(pil_images)
            outputs = self.model(**inputs)
            
            results = []
            for i, (image, original_size) in enumerate(zip(images, original_sizes)):
                try:
                    alpha = self._extract_alpha(outputs, original_size, batch_index=i)
                except Exception as e:
                    logger.warning(f"Batch extraction failed for image {i}, falling back: {e}")
                    single_result = self.process_single(image)
                    results.append(single_result)
                    continue
                
                rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = image
                rgba[:, :, 3] = alpha
                results.append(rgba)
            
            return results
            
        except Exception as e:
            logger.warning(f"Batch processing failed, falling back to single: {e}")
            return [self.process_single(img) for img in images]
    
    def unload(self) -> None:
        """Unload the model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("RMBG model unloaded")
