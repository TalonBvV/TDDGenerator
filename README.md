# Text Detection Dataset Generator

A dataset-agnostic engine that consumes YOLOv8-Seg text detection datasets, extracts/generates text assets, and synthesizes new training data with diverse augmentations.

## Features

- **Dataset-agnostic ingestion**: Accepts any YOLOv8-Seg formatted datasets
- **Real word extraction**: Extracts text crops with alpha matting using RMBG-2.0
- **Synthetic text generation**: Renders text with fonts, effects, and styling
- **Background cleaning**: Removes text from images using LaMa inpainting
- **Advanced composition**: Places text with warping, blending, and collision avoidance
- **Comprehensive augmentations**: Photometric, motion, scene, and document effects
- **Automatic train/val splitting**: Configurable split ratios

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: A100 for best performance)
- ~10GB GPU memory for model inference

## Usage

### Basic Usage

```bash
python -m engine.main \
    --dataset-dir /path/to/datasets/ \
    --wordlist /path/to/words.txt \
    --fonts-dir /path/to/fonts/ \
    --output-dir /path/to/output/
```

### With All Options

```bash
python -m engine.main \
    --dataset-dir /path/to/datasets/ \
    --wordlist /path/to/words.txt \
    --fonts-dir /path/to/fonts/ \
    --output-dir /path/to/output/ \
    --per-sample 5 \
    --real-ratio 0.5 \
    --seed 42 \
    --preview 10 \
    --batch-size 8 \
    --config config.yaml
```

### Using Config File

```bash
python -m engine.main --config my_config.yaml
```

## Input Structure

```
/dataset-dir/
    /dataset_A/
        /images/
            img001.jpg
            img002.jpg
        /labels/
            img001.txt    # YOLOv8-Seg format
            img002.txt
    /dataset_B/
        /images/
        /labels/
    ...

/wordlist.txt           # One word per line
/fonts/
    font1.ttf
    font2.otf
```

### YOLOv8-Seg Label Format

Each line in a label file:
```
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```

Example:
```
0 0.155 0.238 0.201 0.231 0.217 0.289 0.159 0.303
0 0.450 0.120 0.520 0.115 0.525 0.180 0.455 0.185
```

## Output Structure

```
/output/
    /assets/
        /real_text/           # Extracted word assets
        /synth_text/          # Generated synthetic text
        /backgrounds/         # LaMa-cleaned backgrounds
    /preview/                 # Preview samples (if --preview > 0)
    /train/
        /images/
        /labels/
    /val/
        /images/
        /labels/
    dataset.yaml              # YOLO training config
    generation_report.json    # Statistics
```

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-dir` | Required | Root directory with dataset subfolders |
| `--wordlist` | Required | Text file with words (one per line) |
| `--fonts-dir` | Required | Directory with TTF/OTF fonts |
| `--output-dir` | Required | Output directory |
| `--per-sample` | 5 | Alternatives per input image |
| `--real-ratio` | 0.5 | Real vs synthetic asset ratio |
| `--seed` | None | Random seed for reproducibility |
| `--preview` | 0 | Preview samples to generate |
| `--batch-size` | 8 | Batch size for model inference |
| `--device` | cuda | Device (cuda/cpu) |
| `--config` | None | YAML config file |

### YAML Configuration

```yaml
input:
  dataset_dir: /path/to/datasets
  wordlist: /path/to/words.txt
  fonts_dir: /path/to/fonts

output:
  output_dir: /path/to/output
  resolution: [1024, 1024]

generation:
  per_sample: 5
  real_ratio: 0.5
  seed: 42

models:
  lama_checkpoint: big-lama
  rmbg_model: briaai/RMBG-2.0
  batch_size: 8
  device: cuda

text:
  scale_min: 0.03
  scale_max: 0.25

warp:
  types:
    - perspective
    - curve
    - arc
    - sine_wave
    - circular
    - spiral
    - freeform_polygon
  intensity: moderate

blending:
  mode: intermediate
  edge_blur: 1.5
  shadow_opacity: 0.4

augmentation:
  probability: 0.5

preview:
  count: 10

split:
  val_ratio: 0.2
```

## Pipeline Overview

1. **Ingestion**: Scan and validate input datasets
2. **Asset Extraction**: Extract word crops with RMBG alpha matting
3. **Background Cleaning**: Remove text using LaMa inpainting
4. **Preview Generation**: Optional preview samples for inspection
5. **Dataset Generation**: 
   - Compose text onto backgrounds
   - Apply warps (perspective, curve, arc, sine wave, etc.)
   - Blend with color matching and shadows
   - Apply augmentations
6. **Splitting**: Create train/val sets

## Train/Val Split Logic

- **Train**: 100% generated alternatives + 80% original images
- **Val**: 20% original images (held out)

## Training with Generated Dataset

```bash
yolo segment train \
    model=yolov8m-seg.yaml \
    data=/path/to/output/dataset.yaml \
    epochs=100 \
    imgsz=1024
```

## Warp Types

| Type | Description |
|------|-------------|
| perspective | 3D perspective transformation |
| curve | Bezier curve bending |
| arc | Circular arc deformation |
| sine_wave | Sinusoidal wave distortion |
| circular | Cylindrical wrap |
| spiral | Twist/spiral effect |
| freeform_polygon | Random grid displacement |

## Augmentations

**Photometric**: Brightness, contrast, exposure, color temperature, JPEG artifacts

**Motion/Sensor**: Camera shake, Gaussian noise, lens distortion

**Scene**: Fog, rain streaks

**Document**: Paper stains, folds, crumple texture

## License

MIT License
