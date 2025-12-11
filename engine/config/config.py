"""
Configuration management for the dataset generator engine.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class InputConfig:
    dataset_dir: Path
    wordlist: Path
    fonts_dir: Path


@dataclass
class OutputConfig:
    output_dir: Path
    resolution: Tuple[int, int] = (1024, 1024)


@dataclass
class GenerationConfig:
    per_sample: int = 5
    real_ratio: float = 0.5
    seed: Optional[int] = None


@dataclass
class ModelsConfig:
    lama_checkpoint: str = "big-lama"
    rmbg_model: str = "briaai/RMBG-2.0"
    batch_size: int = 8
    device: str = "cuda"


@dataclass
class TextConfig:
    scale_min: float = 0.03
    scale_max: float = 0.25
    scale_bias: str = "medium"


@dataclass
class WarpConfig:
    types: List[str] = field(default_factory=lambda: [
        "perspective", "curve", "arc", "sine_wave", 
        "circular", "spiral", "freeform_polygon"
    ])
    intensity: str = "moderate"


@dataclass
class BlendingConfig:
    mode: str = "intermediate"
    edge_blur: float = 1.5
    shadow_opacity: float = 0.4


@dataclass
class AugmentationConfig:
    probability: float = 0.5


@dataclass
class PreviewConfig:
    count: int = 0


@dataclass
class SplitConfig:
    val_ratio: float = 0.2


@dataclass
class Config:
    input: InputConfig
    output: OutputConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    text: TextConfig = field(default_factory=TextConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    blending: BlendingConfig = field(default_factory=BlendingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Text Detection Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Root directory containing YOLOv8-Seg dataset subfolders"
    )
    parser.add_argument(
        "--wordlist", type=Path, required=True,
        help="Text file with one word per line"
    )
    parser.add_argument(
        "--fonts-dir", type=Path, required=True,
        help="Directory containing TTF/OTF font files"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for generated dataset"
    )
    parser.add_argument(
        "--per-sample", type=int, default=5,
        help="Number of alternative images per input image"
    )
    parser.add_argument(
        "--real-ratio", type=float, default=0.5,
        help="Ratio of real vs synthetic text assets (0.0-1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--preview", type=int, default=0,
        help="Generate N preview samples before full run"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config file (CLI args override YAML)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for LaMa/RMBG inference"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for model inference (cuda/cpu)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint if available"
    )
    
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(yaml_config: dict, args: argparse.Namespace) -> Config:
    """Merge YAML config with CLI arguments (CLI takes precedence)."""
    
    input_cfg = yaml_config.get("input", {})
    output_cfg = yaml_config.get("output", {})
    gen_cfg = yaml_config.get("generation", {})
    models_cfg = yaml_config.get("models", {})
    text_cfg = yaml_config.get("text", {})
    warp_cfg = yaml_config.get("warp", {})
    blend_cfg = yaml_config.get("blending", {})
    aug_cfg = yaml_config.get("augmentation", {})
    preview_cfg = yaml_config.get("preview", {})
    split_cfg = yaml_config.get("split", {})
    
    input_config = InputConfig(
        dataset_dir=args.dataset_dir or Path(input_cfg.get("dataset_dir", "")),
        wordlist=args.wordlist or Path(input_cfg.get("wordlist", "")),
        fonts_dir=args.fonts_dir or Path(input_cfg.get("fonts_dir", ""))
    )
    
    output_config = OutputConfig(
        output_dir=args.output_dir or Path(output_cfg.get("output_dir", "")),
        resolution=tuple(output_cfg.get("resolution", [1024, 1024]))
    )
    
    generation_config = GenerationConfig(
        per_sample=args.per_sample if args.per_sample != 5 else gen_cfg.get("per_sample", 5),
        real_ratio=args.real_ratio if args.real_ratio != 0.5 else gen_cfg.get("real_ratio", 0.5),
        seed=args.seed if args.seed is not None else gen_cfg.get("seed")
    )
    
    models_config = ModelsConfig(
        lama_checkpoint=models_cfg.get("lama_checkpoint", "big-lama"),
        rmbg_model=models_cfg.get("rmbg_model", "briaai/RMBG-2.0"),
        batch_size=args.batch_size if args.batch_size != 8 else models_cfg.get("batch_size", 8),
        device=args.device if args.device != "cuda" else models_cfg.get("device", "cuda")
    )
    
    text_config = TextConfig(
        scale_min=text_cfg.get("scale_min", 0.03),
        scale_max=text_cfg.get("scale_max", 0.25),
        scale_bias=text_cfg.get("scale_bias", "medium")
    )
    
    warp_config = WarpConfig(
        types=warp_cfg.get("types", [
            "perspective", "curve", "arc", "sine_wave",
            "circular", "spiral", "freeform_polygon"
        ]),
        intensity=warp_cfg.get("intensity", "moderate")
    )
    
    blending_config = BlendingConfig(
        mode=blend_cfg.get("mode", "intermediate"),
        edge_blur=blend_cfg.get("edge_blur", 1.5),
        shadow_opacity=blend_cfg.get("shadow_opacity", 0.4)
    )
    
    augmentation_config = AugmentationConfig(
        probability=aug_cfg.get("probability", 0.5)
    )
    
    preview_config = PreviewConfig(
        count=args.preview if args.preview != 0 else preview_cfg.get("count", 0)
    )
    
    split_config = SplitConfig(
        val_ratio=split_cfg.get("val_ratio", 0.2)
    )
    
    return Config(
        input=input_config,
        output=output_config,
        generation=generation_config,
        models=models_config,
        text=text_config,
        warp=warp_config,
        blending=blending_config,
        augmentation=augmentation_config,
        preview=preview_config,
        split=split_config
    )


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """Load configuration from CLI args and optional YAML file."""
    if args is None:
        args = parse_args()
    
    if args.config and args.config.exists():
        yaml_config = load_yaml_config(args.config)
    else:
        yaml_config = {}
    
    return merge_configs(yaml_config, args)
