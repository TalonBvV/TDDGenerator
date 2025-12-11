"""
Main orchestration module for the Text Detection Dataset Generator.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .config import Config, load_config, parse_args
from .models.lama_loader import LamaModel
from .models.rmbg_loader import RMBGModel
from .modules.augment import Augmentor
from .modules.compose import Composer
from .modules.extract_assets import AssetExtractor, AssetLibrary
from .modules.ingest import DatasetIngester, DatasetInfo, SamplePair
from .modules.inpaint_lama import LamaInpainter
from .modules.label_writer import LabelWriter, create_dataset_yaml
from .modules.split import DatasetSplitter
from .modules.synth_generator import SynthGenerator
from .utils import (
    batch_iterator,
    load_image,
    resize_image,
    crop_to_square,
    save_image,
    save_report,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("engine")


class DatasetGenerator:
    """Main dataset generation engine with checkpoint/resume support."""
    
    CHECKPOINT_FILE = "checkpoint.json"
    
    def __init__(self, config: Config, resume: bool = False):
        self.config = config
        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        
        self.logger = setup_logging(self.output_dir)
        
        if config.generation.seed is not None:
            set_seed(config.generation.seed)
            self.logger.info(f"Random seed set to {config.generation.seed}")
        
        self.rmbg_model: Optional[RMBGModel] = None
        self.lama_model: Optional[LamaModel] = None
        
        self.datasets: List[DatasetInfo] = []
        self.real_assets: Optional[AssetLibrary] = None
        self.backgrounds: Dict[str, List[Path]] = {}
        self.wordlist: List[str] = []
        
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        self.completed_samples: set = set()
        
        self.stats = {
            "start_time": None,
            "end_time": None,
            "datasets_processed": 0,
            "real_assets_extracted": 0,
            "synth_assets_generated": 0,
            "backgrounds_cleaned": 0,
            "images_generated": 0,
            "images_failed": 0,
            "total_polygons": 0,
        }
    
    def save_checkpoint(self, stage: str, sample_idx: int = 0) -> None:
        """Save checkpoint for resume capability."""
        checkpoint = {
            "stage": stage,
            "sample_idx": sample_idx,
            "completed_samples": list(self.completed_samples),
            "stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        if not self.checkpoint_path.exists():
            return None
        try:
            with open(self.checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            self.completed_samples = set(checkpoint.get("completed_samples", []))
            self.stats.update(checkpoint.get("stats", {}))
            self.logger.info(f"Loaded checkpoint from stage '{checkpoint['stage']}' "
                           f"with {len(self.completed_samples)} completed samples")
            return checkpoint
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            self.logger.info("Checkpoint cleared")
    
    def load_models(self) -> None:
        """Load ML models (RMBG and LaMa)."""
        self.logger.info("Loading models...")
        
        self.rmbg_model = RMBGModel(
            model_name=self.config.models.rmbg_model,
            device=self.config.models.device
        )
        self.rmbg_model.load()
        
        self.lama_model = LamaModel(
            checkpoint=self.config.models.lama_checkpoint,
            device=self.config.models.device
        )
        self.lama_model.load()
        
        self.logger.info("Models loaded successfully")
    
    def load_wordlist(self) -> None:
        """Load wordlist from file."""
        wordlist_path = Path(self.config.input.wordlist)
        
        if not wordlist_path.exists():
            raise FileNotFoundError(f"Wordlist not found: {wordlist_path}")
        
        with open(wordlist_path, "r", encoding="utf-8") as f:
            self.wordlist = [line.strip() for line in f if line.strip()]
        
        self.logger.info(f"Loaded {len(self.wordlist)} words from wordlist")
    
    def ingest_datasets(self) -> None:
        """Scan and ingest input datasets."""
        self.logger.info("Ingesting datasets...")
        
        ingester = DatasetIngester(self.config.input.dataset_dir)
        self.datasets = ingester.scan()
        
        report = ingester.validate()
        
        if not report.valid:
            raise ValueError(f"Dataset validation failed: {report.errors}")
        
        if report.warnings:
            for warning in report.warnings[:10]:
                self.logger.warning(warning)
        
        self.stats["datasets_processed"] = len(self.datasets)
        self.logger.info(
            f"Ingested {len(self.datasets)} datasets with "
            f"{report.total_images} images and {report.total_polygons} polygons"
        )
    
    def extract_assets(self) -> None:
        """Extract real word assets from datasets."""
        self.logger.info("Extracting word assets...")
        
        extractor = AssetExtractor(
            rmbg_model=self.rmbg_model,
            output_dir=self.output_dir,
            batch_size=self.config.models.batch_size
        )
        
        self.real_assets = extractor.process_all_datasets(
            self.datasets,
            save_assets=True
        )
        
        self.stats["real_assets_extracted"] = self.real_assets.total_count
        self.logger.info(f"Extracted {self.real_assets.total_count} word assets")
    
    def clean_backgrounds(self) -> None:
        """Clean backgrounds using LaMa inpainting."""
        self.logger.info("Cleaning backgrounds...")
        
        inpainter = LamaInpainter(
            lama_model=self.lama_model,
            output_dir=self.output_dir,
            batch_size=self.config.models.batch_size
        )
        
        self.backgrounds = inpainter.process_all_datasets(
            self.datasets,
            save_backgrounds=True
        )
        
        total_bgs = sum(len(paths) for paths in self.backgrounds.values())
        self.stats["backgrounds_cleaned"] = total_bgs
        self.logger.info(f"Cleaned {total_bgs} backgrounds")
    
    def generate_preview(self) -> None:
        """Generate preview samples."""
        if self.config.preview.count <= 0:
            return
        
        self.logger.info(f"Generating {self.config.preview.count} preview samples...")
        
        preview_dir = self.output_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        synth_generator = SynthGenerator(
            fonts_dir=self.config.input.fonts_dir,
            wordlist=self.wordlist,
            output_dir=self.output_dir
        )
        
        composer = Composer(
            output_size=self.config.output.resolution,
            real_ratio=self.config.generation.real_ratio,
            scale_min=self.config.text.scale_min,
            scale_max=self.config.text.scale_max,
            edge_blur=self.config.blending.edge_blur,
            shadow_opacity=self.config.blending.shadow_opacity
        )
        
        augmentor = Augmentor(probability=self.config.augmentation.probability)
        label_writer = LabelWriter(self.output_dir)
        
        all_backgrounds = []
        for dataset_name, bg_paths in self.backgrounds.items():
            for bg_path in bg_paths:
                all_backgrounds.append((dataset_name, bg_path))
        
        if not all_backgrounds:
            self.logger.warning("No backgrounds available for preview")
            return
        
        all_samples = []
        for dataset in self.datasets:
            for sample in dataset.samples:
                all_samples.append((dataset.name, sample))
        
        for i in range(min(self.config.preview.count, len(all_samples))):
            try:
                dataset_name, sample = all_samples[i % len(all_samples)]
                
                bg_dataset, bg_path = all_backgrounds[i % len(all_backgrounds)]
                background = load_image(bg_path)
                
                result = composer.compose(
                    background=background,
                    original_polygons=sample.polygons,
                    real_assets=self.real_assets,
                    synth_generator=synth_generator,
                    fill_random=True
                )
                
                aug_image, aug_polygons = augmentor.augment(
                    result.image, result.polygons
                )
                
                preview_image_path = preview_dir / f"preview_{i:04d}.png"
                preview_label_path = preview_dir / f"preview_{i:04d}.txt"
                
                save_image(aug_image, preview_image_path)
                label_writer.write(aug_polygons, preview_label_path)
                
            except Exception as e:
                self.logger.error(f"Failed to generate preview {i}: {e}")
        
        self.logger.info(f"Preview samples saved to {preview_dir}")
    
    def generate_dataset(self) -> None:
        """Generate the full dataset."""
        self.logger.info("Starting dataset generation...")
        
        gen_dir = self.output_dir / "generated"
        gen_images_dir = gen_dir / "images"
        gen_labels_dir = gen_dir / "labels"
        gen_images_dir.mkdir(parents=True, exist_ok=True)
        gen_labels_dir.mkdir(parents=True, exist_ok=True)
        
        synth_generator = SynthGenerator(
            fonts_dir=self.config.input.fonts_dir,
            wordlist=self.wordlist,
            output_dir=self.output_dir
        )
        
        composer = Composer(
            output_size=self.config.output.resolution,
            real_ratio=self.config.generation.real_ratio,
            scale_min=self.config.text.scale_min,
            scale_max=self.config.text.scale_max,
            edge_blur=self.config.blending.edge_blur,
            shadow_opacity=self.config.blending.shadow_opacity
        )
        
        augmentor = Augmentor(probability=self.config.augmentation.probability)
        label_writer = LabelWriter(self.output_dir)
        
        all_backgrounds = []
        bg_weights = []
        for dataset_name, bg_paths in self.backgrounds.items():
            weight = 1.0 / max(len(bg_paths), 1)
            for bg_path in bg_paths:
                all_backgrounds.append((dataset_name, bg_path))
                bg_weights.append(weight)
        
        if bg_weights:
            total_weight = sum(bg_weights)
            bg_weights = [w / total_weight for w in bg_weights]
        
        all_samples = []
        for dataset in self.datasets:
            for sample in dataset.samples:
                all_samples.append((dataset.name, sample))
        
        total_to_generate = len(all_samples) * self.config.generation.per_sample
        self.logger.info(
            f"Generating {total_to_generate} images "
            f"({len(all_samples)} samples × {self.config.generation.per_sample} alternatives)"
        )
        
        original_samples = []
        alternative_samples = []
        
        pbar = tqdm(total=total_to_generate, desc="Generating images")
        
        checkpoint_interval = 50
        
        for sample_idx, (dataset_name, sample) in enumerate(all_samples):
            sample_key = f"{dataset_name}_{sample.image_path.stem}"
            
            orig_image = load_image(sample.image_path)
            orig_image = crop_to_square(orig_image)
            orig_image = resize_image(orig_image, self.config.output.resolution)
            
            orig_image_path = gen_images_dir / f"orig_{sample.image_path.stem}.png"
            orig_label_path = gen_labels_dir / f"orig_{sample.image_path.stem}.txt"
            
            if not orig_image_path.exists():
                save_image(orig_image, orig_image_path)
            
            h, w = self.config.output.resolution
            scaled_polygons = []
            for polygon in sample.polygons:
                scaled_polygons.append(polygon)
            
            if not orig_label_path.exists():
                label_writer.write(scaled_polygons, orig_label_path)
            original_samples.append((orig_image_path, orig_label_path))
            
            for alt_idx in range(self.config.generation.per_sample):
                alt_key = f"{sample_key}_{alt_idx}"
                
                if alt_key in self.completed_samples:
                    alt_image_path = gen_images_dir / f"alt_{sample.image_path.stem}_{alt_idx:03d}.png"
                    alt_label_path = gen_labels_dir / f"alt_{sample.image_path.stem}_{alt_idx:03d}.txt"
                    if alt_image_path.exists() and alt_label_path.exists():
                        alternative_samples.append((alt_image_path, alt_label_path))
                    pbar.update(1)
                    continue
                
                try:
                    if all_backgrounds:
                        bg_idx = np.random.choice(len(all_backgrounds), p=bg_weights)
                        _, bg_path = all_backgrounds[bg_idx]
                        background = load_image(bg_path)
                    else:
                        background = orig_image.copy()
                    
                    result = composer.compose(
                        background=background,
                        original_polygons=sample.polygons,
                        real_assets=self.real_assets,
                        synth_generator=synth_generator,
                        fill_random=True
                    )
                    
                    aug_image, aug_polygons = augmentor.augment(
                        result.image, result.polygons
                    )
                    
                    alt_image_path = gen_images_dir / f"alt_{sample.image_path.stem}_{alt_idx:03d}.png"
                    alt_label_path = gen_labels_dir / f"alt_{sample.image_path.stem}_{alt_idx:03d}.txt"
                    
                    save_image(aug_image, alt_image_path)
                    label_writer.write(aug_polygons, alt_label_path)
                    
                    alternative_samples.append((alt_image_path, alt_label_path))
                    self.completed_samples.add(alt_key)
                    
                    self.stats["images_generated"] += 1
                    self.stats["total_polygons"] += len(aug_polygons)
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate alternative {alt_idx} for "
                        f"{sample.image_path.stem}: {e}"
                    )
                    self.stats["images_failed"] += 1
                
                pbar.update(1)
            
            if (sample_idx + 1) % checkpoint_interval == 0:
                self.save_checkpoint("generation", sample_idx)
        
        pbar.close()
        
        self.logger.info("Splitting dataset into train/val...")
        
        splitter = DatasetSplitter(
            output_dir=self.output_dir,
            val_ratio=self.config.split.val_ratio,
            seed=self.config.generation.seed
        )
        
        split_stats = splitter.split(original_samples, alternative_samples)
        
        self.stats.update(split_stats)
        
        create_dataset_yaml(self.output_dir)
        
        self.logger.info("Dataset generation complete")
    
    def save_report(self) -> None:
        """Save generation report."""
        self.stats["end_time"] = datetime.now().isoformat()
        
        if self.stats["start_time"]:
            start = datetime.fromisoformat(self.stats["start_time"])
            end = datetime.fromisoformat(self.stats["end_time"])
            duration = (end - start).total_seconds()
            
            self.stats["duration_seconds"] = duration
            self.stats["images_per_second"] = (
                self.stats["images_generated"] / duration if duration > 0 else 0
            )
        
        report = {
            "timestamp": self.stats["end_time"],
            "config": {
                "dataset_dir": str(self.config.input.dataset_dir),
                "output_dir": str(self.config.output.output_dir),
                "per_sample": self.config.generation.per_sample,
                "real_ratio": self.config.generation.real_ratio,
                "seed": self.config.generation.seed,
                "resolution": self.config.output.resolution,
            },
            "input_stats": {
                "datasets": self.stats["datasets_processed"],
                "total_images": sum(ds.num_samples for ds in self.datasets),
                "total_polygons": sum(ds.num_polygons for ds in self.datasets),
            },
            "asset_stats": {
                "real_words_extracted": self.stats["real_assets_extracted"],
                "backgrounds_cleaned": self.stats["backgrounds_cleaned"],
            },
            "output_stats": {
                "images_generated": self.stats["images_generated"],
                "images_failed": self.stats["images_failed"],
                "total_polygons_placed": self.stats["total_polygons"],
                "train_images": self.stats.get("total_train", 0),
                "val_images": self.stats.get("total_val", 0),
            },
            "performance": {
                "duration_seconds": self.stats.get("duration_seconds", 0),
                "images_per_second": self.stats.get("images_per_second", 0),
            }
        }
        
        report_path = self.output_dir / "generation_report.json"
        save_report(report, report_path)
        
        self.logger.info(f"Report saved to {report_path}")
        
        print("\n" + "=" * 50)
        print("GENERATION SUMMARY")
        print("=" * 50)
        print(f"Datasets processed: {report['input_stats']['datasets']}")
        print(f"Real assets extracted: {report['asset_stats']['real_words_extracted']}")
        print(f"Backgrounds cleaned: {report['asset_stats']['backgrounds_cleaned']}")
        print(f"Images generated: {report['output_stats']['images_generated']}")
        print(f"Images failed: {report['output_stats']['images_failed']}")
        print(f"Train set: {report['output_stats']['train_images']} images")
        print(f"Val set: {report['output_stats']['val_images']} images")
        print(f"Total time: {report['performance']['duration_seconds']:.1f} seconds")
        print(f"Speed: {report['performance']['images_per_second']:.2f} images/sec")
        print("=" * 50)
    
    def run(self) -> None:
        """Run the full generation pipeline with checkpoint/resume support."""
        checkpoint = None
        if self.resume:
            checkpoint = self.load_checkpoint()
        
        if not self.stats.get("start_time"):
            self.stats["start_time"] = datetime.now().isoformat()
        
        start_stage = checkpoint.get("stage", "init") if checkpoint else "init"
        stages = ["init", "wordlist", "models", "ingest", "extract", "backgrounds", 
                  "preview", "generation", "split", "done"]
        
        try:
            start_idx = stages.index(start_stage) if start_stage in stages else 0
            
            if start_idx <= stages.index("wordlist"):
                self.load_wordlist()
                self.save_checkpoint("wordlist")
            
            if start_idx <= stages.index("models"):
                self.load_models()
                self.save_checkpoint("models")
            
            if start_idx <= stages.index("ingest"):
                self.ingest_datasets()
                self.save_checkpoint("ingest")
            
            if start_idx <= stages.index("extract"):
                self.extract_assets()
                self.save_checkpoint("extract")
            
            if start_idx <= stages.index("backgrounds"):
                self.clean_backgrounds()
                self.save_checkpoint("backgrounds")
            
            if start_idx <= stages.index("preview"):
                self.generate_preview()
                self.save_checkpoint("preview")
            
            if start_idx <= stages.index("generation"):
                self.generate_dataset()
                self.save_checkpoint("generation")
            
            self.save_report()
            self.clear_checkpoint()
            
        except KeyboardInterrupt:
            self.logger.warning("Generation interrupted by user. Progress saved to checkpoint.")
            self.save_checkpoint("generation")
            raise
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            self.save_checkpoint("generation")
            raise
        
        finally:
            if self.rmbg_model:
                self.rmbg_model.unload()
            if self.lama_model:
                self.lama_model.unload()


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args)
    
    resume = getattr(args, 'resume', False)
    generator = DatasetGenerator(config, resume=resume)
    generator.run()


if __name__ == "__main__":
    main()
