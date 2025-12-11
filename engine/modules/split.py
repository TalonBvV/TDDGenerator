"""
Dataset splitting module for creating train/val splits.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("engine.split")


class DatasetSplitter:
    """Splits generated dataset into train and validation sets."""
    
    def __init__(
        self,
        output_dir: Path,
        val_ratio: float = 0.2,
        seed: int = None
    ):
        """
        Initialize dataset splitter.
        
        Args:
            output_dir: Output directory for split datasets
            val_ratio: Ratio of original images for validation (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.val_ratio = val_ratio
        self.seed = seed
        
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        
        self.train_images_dir = self.train_dir / "images"
        self.train_labels_dir = self.train_dir / "labels"
        self.val_images_dir = self.val_dir / "images"
        self.val_labels_dir = self.val_dir / "labels"
    
    def setup_directories(self) -> None:
        """Create output directories."""
        for dir_path in [
            self.train_images_dir,
            self.train_labels_dir,
            self.val_images_dir,
            self.val_labels_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def split(
        self,
        original_samples: List[Tuple[Path, Path]],
        alternative_samples: List[Tuple[Path, Path]]
    ) -> Dict[str, int]:
        """
        Split samples into train and validation sets.
        
        Split logic:
        - Train = 100% alternatives + 80% originals
        - Val = 20% originals
        
        Args:
            original_samples: List of (image_path, label_path) for original images
            alternative_samples: List of (image_path, label_path) for generated alternatives
            
        Returns:
            Dictionary with counts for each split
        """
        self.setup_directories()
        
        if self.seed is not None:
            random.seed(self.seed)
        
        shuffled_originals = original_samples.copy()
        random.shuffle(shuffled_originals)
        
        val_count = int(len(shuffled_originals) * self.val_ratio)
        val_originals = shuffled_originals[:val_count]
        train_originals = shuffled_originals[val_count:]
        
        stats = {
            "train_alternatives": 0,
            "train_originals": 0,
            "val_originals": 0,
            "total_train": 0,
            "total_val": 0
        }
        
        logger.info("Copying alternative samples to train set...")
        for img_path, label_path in alternative_samples:
            self._copy_sample(
                img_path, label_path,
                self.train_images_dir, self.train_labels_dir
            )
            stats["train_alternatives"] += 1
        
        logger.info("Copying original samples to train set...")
        for img_path, label_path in train_originals:
            self._copy_sample(
                img_path, label_path,
                self.train_images_dir, self.train_labels_dir
            )
            stats["train_originals"] += 1
        
        logger.info("Copying original samples to val set...")
        for img_path, label_path in val_originals:
            self._copy_sample(
                img_path, label_path,
                self.val_images_dir, self.val_labels_dir
            )
            stats["val_originals"] += 1
        
        stats["total_train"] = stats["train_alternatives"] + stats["train_originals"]
        stats["total_val"] = stats["val_originals"]
        
        logger.info(
            f"Split complete: Train={stats['total_train']} "
            f"({stats['train_alternatives']} alt + {stats['train_originals']} orig), "
            f"Val={stats['total_val']}"
        )
        
        return stats
    
    def _copy_sample(
        self,
        src_image: Path,
        src_label: Path,
        dst_images_dir: Path,
        dst_labels_dir: Path
    ) -> None:
        """Copy a sample (image + label) to destination."""
        dst_image = dst_images_dir / src_image.name
        dst_label = dst_labels_dir / src_label.name
        
        counter = 0
        while dst_image.exists():
            counter += 1
            stem = src_image.stem
            suffix = src_image.suffix
            dst_image = dst_images_dir / f"{stem}_{counter}{suffix}"
            dst_label = dst_labels_dir / f"{stem}_{counter}.txt"
        
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)
    
    def split_in_place(
        self,
        generated_dir: Path,
        originals_info: List[Tuple[str, str]]
    ) -> Dict[str, int]:
        """
        Split when generated images are already in a directory.
        
        Args:
            generated_dir: Directory containing generated images/labels
            originals_info: List of (image_name, label_name) for original samples
            
        Returns:
            Dictionary with counts for each split
        """
        self.setup_directories()
        
        if self.seed is not None:
            random.seed(self.seed)
        
        gen_images_dir = generated_dir / "images"
        gen_labels_dir = generated_dir / "labels"
        
        original_names = set(name for name, _ in originals_info)
        
        all_images = list(gen_images_dir.glob("*"))
        
        originals = []
        alternatives = []
        
        for img_path in all_images:
            label_path = gen_labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            is_original = any(
                img_path.stem == name or img_path.stem.startswith(f"{name}_")
                for name in original_names
            )
            
            if is_original and not any(
                c.isdigit() for c in img_path.stem.split("_")[-1]
            ):
                originals.append((img_path, label_path))
            else:
                alternatives.append((img_path, label_path))
        
        return self.split(originals, alternatives)
    
    def get_stats(self) -> Dict[str, int]:
        """Get current split statistics."""
        train_images = len(list(self.train_images_dir.glob("*")))
        train_labels = len(list(self.train_labels_dir.glob("*.txt")))
        val_images = len(list(self.val_images_dir.glob("*")))
        val_labels = len(list(self.val_labels_dir.glob("*.txt")))
        
        return {
            "train_images": train_images,
            "train_labels": train_labels,
            "val_images": val_images,
            "val_labels": val_labels
        }
